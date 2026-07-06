#!/usr/bin/env python3
"""
WC2026 game-by-game bracket predictor.
Predicts every remaining match from group stage through the final.

Usage:
  PYTHONPATH=/Users/ssrotriyo/world_cup_2026 python3 src/predict_bracket.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from src.model import load_model, build_match_features, _apply_calibration, MATCH_FEATURES, WC2026_SEEDED_ELO
from src.features import HOST_COUNTRY_LEAGUES
from src.ingest.live_results import (
    COMPLETED_MATCHES, COMPLETED_KNOCKOUT, WC2026_ACTUAL_GROUPS,
    compute_standings, get_group_rank, compute_form_z,
)

# ---------------------------------------------------------------------------
# Host familiarity boost
# WC2026 is hosted in USA/MEX/CAN; players active in those leagues know the
# venues, travel patterns, and conditions. Applied as a post-prediction
# probability shift so the model (trained on non-host WCs) isn't asked to
# extrapolate an effect it never observed in training data.
# ---------------------------------------------------------------------------
MAX_HOST_FAMILIARITY_BOOST = 0.04  # max 4-percentage-point shift
MAX_FORM_BOOST = 0.06              # max 6-percentage-point shift from in-tournament form
FORM_SCALE = 0.02                  # per z-unit of opponent-adjusted form difference
FORM_Z = compute_form_z(WC2026_SEEDED_ELO)  # live form, computed once


def _get_host_familiarity(team: str, country: str) -> float:
    """Return team's position-weighted fraction of squad in the host country's league."""
    col = f"host_familiarity_{country}"
    if team not in team_features["team"].values or col not in team_features.columns:
        return 0.0
    row = team_features.loc[team_features["team"] == team, col]
    return float(row.iloc[0]) if not row.empty else 0.0

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
print("Loading model...", file=sys.stderr)
model, le, feature_cols, elo_base, calibrators = load_model()
team_features = pd.read_parquet("data/processed/team_features.parquet")

# ---------------------------------------------------------------------------
# ELO: update from completed matches
# ---------------------------------------------------------------------------
K = 64

def elo_expected(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def update_elo(ratings: dict, home: str, away: str, hg: int, ag: int) -> dict:
    r = dict(ratings)
    ra, rb = r.get(home, 1500), r.get(away, 1500)
    ea = elo_expected(ra, rb)
    s = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    r[home] = ra + K * (s - ea)
    r[away] = rb + K * ((1 - s) - (1 - ea))
    return r

# Seed with the pre-tournament ELO priors (worldcupelo.com) for 2026 teams — same
# basis the 500k simulation uses — so the deterministic bracket and the sim agree.
elo_live = {**elo_base, **WC2026_SEEDED_ELO}
for m in COMPLETED_MATCHES:
    elo_live = update_elo(elo_live, m["home"], m["away"], m["hg"], m["ag"])
# Apply completed knockout results to ELO, and build a winner-lookup that locks
# those matches in the bracket (actual winner advances, not the model favourite).
for m in COMPLETED_KNOCKOUT:
    elo_live = update_elo(elo_live, m["home"], m["away"], m["hg"], m["ag"])
KO_WINNER = {frozenset((m["home"], m["away"])): m["winner"] for m in COMPLETED_KNOCKOUT}

# ---------------------------------------------------------------------------
# Core predict function
# ---------------------------------------------------------------------------

def predict(
    home: str,
    away: str,
    is_knockout: bool = False,
    elo: dict = None,
    host_country: str = "usa",
) -> dict:
    """
    Returns {"home_win": p, "draw": p, "away_win": p, "home_advance": p}
    home_advance = home_win in group stage; home_win + 1/2*(draw AET) in KO.

    host_country: "usa" | "mex" | "can" — which WC2026 host the match is in.
    """
    if elo is None:
        elo = elo_live

    # Lock completed knockout results: the team that actually advanced wins 100%.
    ko = KO_WINNER.get(frozenset((home, away)))
    if ko is not None:
        hw = 1.0 if ko == home else 0.0
        return {"home_win": hw, "draw": 0.0, "away_win": 1.0 - hw,
                "home_advance": hw, "away_advance": 1.0 - hw}

    stage_name = "round of 16" if is_knockout else ""
    dummy = pd.DataFrame([{
        "home_team_name": home,
        "away_team_name": away,
        "home_score": np.nan,
        "away_score": np.nan,
        "tournament_name": "wc_2026",
        "neutral": True,
        "competition_stage_name": stage_name,
    }])

    match_df = build_match_features(dummy, team_features, elo)
    if match_df.empty:
        p = 1 / 3
        return {"home_win": p, "draw": p, "away_win": p,
                "home_advance": p + p * 0.5}

    X = match_df[[c for c in feature_cols if c in match_df.columns]] \
        .apply(pd.to_numeric, errors="coerce").fillna(0).values

    raw = model.predict_proba(X)
    cal = _apply_calibration(raw, calibrators)[0]
    classes = le.classes_  # ['away_win','draw','home_win']
    d = {cls: float(p) for cls, p in zip(classes, cal)}

    hw = d.get("home_win", 1/3)
    draw = d.get("draw", 1/3)
    aw = d.get("away_win", 1/3)

    # Host familiarity post-prediction adjustment.
    # Shift home_win probability toward the team whose players are more active
    # in the host country's domestic league.
    if host_country:
        fam_h = _get_host_familiarity(home, host_country)
        fam_a = _get_host_familiarity(away, host_country)
        fam_diff = fam_h - fam_a  # positive = home team more familiar
        boost = np.clip(fam_diff * MAX_HOST_FAMILIARITY_BOOST, -MAX_HOST_FAMILIARITY_BOOST, MAX_HOST_FAMILIARITY_BOOST)
        if boost != 0:
            # Shift probability mass from loser to winner; draw stays constant.
            if boost > 0:
                transfer = min(boost, aw)  # can't take more than away_win has
                hw = hw + transfer
                aw = aw - transfer
            else:
                transfer = min(-boost, hw)
                aw = aw + transfer
                hw = hw - transfer
        # Re-normalise to [0,1]
        total = hw + draw + aw
        if total > 0:
            hw, draw, aw = hw / total, draw / total, aw / total

    # In-tournament form adjustment — opponent-adjusted GD/game, z-scored.
    # Live evidence the static squad features can't see (a star's club form not
    # translating; a hot scorer like Messi). Capped so 3 games can't dominate.
    fz_diff = FORM_Z.get(home, 0.0) - FORM_Z.get(away, 0.0)
    fshift = float(np.clip(fz_diff * FORM_SCALE, -MAX_FORM_BOOST, MAX_FORM_BOOST))
    if fshift > 0:
        transfer = min(fshift, aw); hw += transfer; aw -= transfer
    elif fshift < 0:
        transfer = min(-fshift, hw); aw += transfer; hw -= transfer
    total = hw + draw + aw
    if total > 0:
        hw, draw, aw = hw / total, draw / total, aw / total

    # In knockout, draw leads to extra time + penalties; model draw ~= 50/50 advance
    if is_knockout:
        home_adv = hw + draw * 0.5
    else:
        home_adv = hw  # unused for group stage

    return {
        "home_win": round(hw, 4),
        "draw": round(draw, 4),
        "away_win": round(aw, 4),
        "home_advance": round(home_adv, 4),
        "away_advance": round(1 - (hw + draw * 0.5), 4) if is_knockout else round(aw, 4),
    }

# ---------------------------------------------------------------------------
# Predict remaining group stage games
# ---------------------------------------------------------------------------

REMAINING = [
    # Group J — June 27
    {"group": "J", "home": "Algeria",    "away": "Austria"},
    {"group": "J", "home": "Jordan",     "away": "Argentina"},
    # Group K — June 27
    {"group": "K", "home": "Colombia",   "away": "Portugal"},
    {"group": "K", "home": "DR Congo",   "away": "Uzbekistan"},
    # Group L — June 27
    {"group": "L", "home": "Panama",     "away": "England"},
    {"group": "L", "home": "Croatia",    "away": "Ghana"},
]

print("Predicting group stage games...", file=sys.stderr)

group_preds = []
for m in REMAINING:
    p = predict(m["home"], m["away"], is_knockout=False)
    group_preds.append({
        "group": m["group"],
        "home": m["home"],
        "away": m["away"],
        **p,
        "predicted_winner": m["home"] if p["home_win"] > p["away_win"] else
                            (m["away"] if p["away_win"] > p["home_win"] else "draw"),
    })

# ---------------------------------------------------------------------------
# Simulate expected group outcomes to determine qualifiers
# ---------------------------------------------------------------------------

def expected_outcome(hw, draw, aw):
    """Most likely outcome: returns ('home','draw','away')"""
    if hw >= draw and hw >= aw:
        return "home"
    if aw >= draw and aw >= hw:
        return "away"
    return "draw"

# Build expected simulated standings
# Start from current completed matches
standings_base = compute_standings(COMPLETED_MATCHES)

# Add expected results for remaining games
expected_added = list(COMPLETED_MATCHES)
for m in group_preds:
    outcome = expected_outcome(m["home_win"], m["draw"], m["away_win"])
    if outcome == "home":
        hg, ag = 1, 0
    elif outcome == "away":
        hg, ag = 0, 1
    else:
        hg, ag = 1, 1
    expected_added.append({
        "group": m["group"], "home": m["home"], "away": m["away"],
        "hg": hg, "ag": ag, "date": "predicted",
    })

final_standings = compute_standings(expected_added)

# Rank each group
group_ranked = {}
for grp, teams in WC2026_ACTUAL_GROUPS.items():
    group_ranked[grp] = get_group_rank(final_standings.get(grp, {}), teams)

# Determine top-2 qualifiers per group
qualifiers = {}  # grp -> [1st, 2nd]
for grp, ranked in group_ranked.items():
    qualifiers[grp] = ranked[:2]

# 3rd-place teams and their points (for the 8 best 3rd-place)
third_place = []
for grp, ranked in group_ranked.items():
    if len(ranked) >= 3:
        t = ranked[2]
        s = final_standings.get(grp, {}).get(t, {"pts": 0, "gd": 0, "gf": 0})
        third_place.append((t, grp, s["pts"], s["gd"], s["gf"]))

third_place_sorted = sorted(third_place, key=lambda x: (-x[2], -x[3], -x[4]))
best_thirds = [t[0] for t in third_place_sorted[:8]]

# ---------------------------------------------------------------------------
# R32 bracket (official WC2026 assignment)
# ---------------------------------------------------------------------------
# R32 matches M73–M88 per FIFA/FotMob bracket
# Format: (slot, description, home_source, away_source)
# Sources: W_X = winner of group X, R_X = runner-up of group X, T_* = best 3rd

# The 8 best 3rd-place teams are assigned to specific R32 slots depending on
# which groups they come from. For simplicity, we use the FIFA 2026 allocation
# table (the 8-best-third allocation is determined by a FIFA matrix at the
# end of the group stage). We'll use a simplified assignment here.

# Official R32 match structure:
R32_DEFS = [
    ("M73", "R_A",  "R_B"),
    ("M74", "W_E",  "T_ABCDF"),   # best 3rd from groups A,B,C,D,F
    ("M75", "W_F",  "R_C"),
    ("M76", "W_C",  "R_F"),
    ("M77", "W_I",  "T_CDFGH"),
    ("M78", "R_E",  "R_I"),
    ("M79", "W_A",  "T_CEFHI"),
    ("M80", "W_L",  "T_EHIJK"),
    ("M81", "W_D",  "T_BEFIJ"),
    ("M82", "W_G",  "T_AEHIJ"),
    ("M83", "R_K",  "R_L"),
    ("M84", "W_H",  "R_J"),
    ("M85", "W_B",  "T_EFGIJ"),
    ("M86", "W_J",  "R_H"),
    ("M87", "W_K",  "T_DEIJL"),
    ("M88", "R_D",  "R_G"),
]

def resolve_slot(slot: str, qualifiers: dict, best_thirds: list) -> str:
    """Resolve a slot code to a team name."""
    if slot.startswith("W_"):
        g = slot[2]
        return qualifiers[g][0]
    if slot.startswith("R_"):
        g = slot[2]
        return qualifiers[g][1]
    if slot.startswith("T_"):
        # Best 3rd assignment: pick from best_thirds in order
        # The letter codes indicate which group's 3rd-place teams are eligible
        # We'll assign the 8 best thirds to the 8 T-slots in order
        return None  # Filled in below

    return None

# Assign 8 best 3rd-place teams to T-slots in order
t_slots = [d for d in R32_DEFS if d[1].startswith("T_") or d[2].startswith("T_")]
thirds_iter = iter(best_thirds)
t_assignments = {}
for d in R32_DEFS:
    for slot in [d[1], d[2]]:
        if slot.startswith("T_") and slot not in t_assignments:
            try:
                t_assignments[slot] = next(thirds_iter)
            except StopIteration:
                t_assignments[slot] = "TBD"

def resolve(slot: str) -> str:
    if slot.startswith("T_"):
        return t_assignments.get(slot, "TBD")
    if slot.startswith("W_"):
        g = slot[2]
        return qualifiers.get(g, ["TBD", "TBD"])[0]
    if slot.startswith("R_"):
        g = slot[2]
        return qualifiers.get(g, ["TBD", "TBD"])[1]
    return slot  # already a team name

# Build R32 matchups
r32_matchups = []
for match_id, home_slot, away_slot in R32_DEFS:
    home = resolve(home_slot)
    away = resolve(away_slot)
    r32_matchups.append({
        "match": match_id,
        "home_slot": home_slot,
        "away_slot": away_slot,
        "home": home,
        "away": away,
    })

# ---------------------------------------------------------------------------
# Predict all knockout rounds
# ---------------------------------------------------------------------------

print("Predicting knockout rounds...", file=sys.stderr)

rounds = []
current_round = r32_matchups

round_names = ["Round of 32", "Round of 16", "Quarter-Final", "Semi-Final", "Final"]

for round_name in round_names:
    round_results = []
    for m in current_round:
        home, away = m["home"], m["away"]
        if home == "TBD" or away == "TBD":
            p = {"home_win": 0.5, "draw": 0.1, "away_win": 0.4,
                 "home_advance": 0.5, "away_advance": 0.5}
        else:
            p = predict(home, away, is_knockout=True)

        predicted_winner = home if p["home_advance"] >= 0.5 else away

        round_results.append({
            "round": round_name,
            "match": m.get("match", ""),
            "home": home,
            "away": away,
            "home_win_pct": round(p["home_win"] * 100, 1),
            "draw_pct": round(p["draw"] * 100, 1),
            "away_win_pct": round(p["away_win"] * 100, 1),
            "home_advance_pct": round(p["home_advance"] * 100, 1),
            "away_advance_pct": round(p["away_advance"] * 100, 1),
            "predicted_winner": predicted_winner,
        })

    rounds.append(round_results)

    if round_name == "Final":
        break

    # Build next round: pair up winners
    winners = [r["predicted_winner"] for r in round_results]
    next_round = []
    for i in range(0, len(winners), 2):
        next_round.append({
            "match": f"predicted_{round_name.replace(' ','')}_{i//2+1}",
            "home": winners[i],
            "away": winners[i + 1] if i + 1 < len(winners) else "TBD",
        })
    current_round = next_round

# ---------------------------------------------------------------------------
# Output JSON for the artifact
# ---------------------------------------------------------------------------
output = {
    "group_predictions": group_preds,
    "group_qualified": {g: qualifiers[g] for g in sorted(qualifiers)},
    "best_thirds": best_thirds,
    "rounds": rounds,
    "champion": rounds[-1][0]["predicted_winner"] if rounds and rounds[-1] else "TBD",
}

print(json.dumps(output, indent=2))
