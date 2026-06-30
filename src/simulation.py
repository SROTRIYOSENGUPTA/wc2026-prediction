"""
Monte Carlo tournament simulation — WC2026 bracket.

CPU mode:  numpy vectorised, ~30s for 100k simulations
GPU mode:  CuPy drop-in replacement, ~2s for 100k simulations on Amaral cluster

Usage:
    python src/simulation.py                  # CPU, 100k sims
    python src/simulation.py --gpu            # GPU via CuPy
    python src/simulation.py --n 500000 --gpu # 500k sims on GPU

WC2026 format:
    48 teams → 8 groups of 6
    Top 2 + 8 best 3rd-place → Round of 32
    Round of 32 → Round of 16 → QF → SF → Final
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import yaml
import json

CFG = yaml.safe_load(open(Path(__file__).parents[1] / "configs/config.yaml"))
PROCESSED_DIR = Path(__file__).parents[1] / CFG["paths"]["processed"]
OUTPUT_DIR = Path(__file__).parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# WC2026 official group draw
# ---------------------------------------------------------------------------

WC2026_GROUPS = {
    "A": ["Mexico",      "South Africa",          "South Korea", "Czechia"],
    "B": ["Switzerland", "Canada",                "Bosnia and Herzegovina", "Qatar"],
    "C": ["Brazil",      "Morocco",               "Scotland",    "Haiti"],
    "D": ["USA",         "Australia",             "Paraguay",    "Turkey"],
    "E": ["Germany",     "Ivory Coast",           "Ecuador",     "Curaçao"],
    "F": ["Netherlands", "Japan",                 "Sweden",      "Tunisia"],
    "G": ["Egypt",       "Iran",                  "Belgium",     "New Zealand"],
    "H": ["Spain",       "Uruguay",               "Cabo Verde",  "Saudi Arabia"],
    "I": ["France",      "Norway",                "Senegal",     "Iraq"],
    "J": ["Argentina",   "Austria",               "Algeria",     "Jordan"],
    "K": ["Colombia",    "Portugal",              "DR Congo",    "Uzbekistan"],
    "L": ["England",     "Ghana",                 "Croatia",     "Panama"],
}

# ---------------------------------------------------------------------------
# Official WC2026 Round of 32 bracket
# Each entry is (slot_a, slot_b) where slot can be:
#   ("W","X")  = winner of group X
#   ("R","X")  = runner-up of group X
#   ("T","X/Y/Z") = best 3rd-place team from groups X, Y, or Z
# Source: Wikipedia — 2026 FIFA World Cup knockout stage
# ---------------------------------------------------------------------------
R32_MATCHES = [
    # M73
    (("R","A"), ("R","B")),
    # M74
    (("W","E"), ("T","A/B/C/D/F")),
    # M75
    (("W","F"), ("R","C")),
    # M76
    (("W","C"), ("R","F")),
    # M77
    (("W","I"), ("T","C/D/F/G/H")),
    # M78
    (("R","E"), ("R","I")),
    # M79
    (("W","A"), ("T","C/E/F/H/I")),
    # M80
    (("W","L"), ("T","E/H/I/J/K")),
    # M81
    (("W","D"), ("T","B/E/F/I/J")),
    # M82
    (("W","G"), ("T","A/E/H/I/J")),
    # M83
    (("R","K"), ("R","L")),
    # M84
    (("W","H"), ("R","J")),
    # M85
    (("W","B"), ("T","E/F/G/I/J")),
    # M86
    (("W","J"), ("R","H")),
    # M87
    (("W","K"), ("T","D/E/I/J/L")),
    # M88
    (("R","D"), ("R","G")),
]

# R16 pairs: winner of M(72+i) vs winner of M(72+i+1), paired as below
# (M73 winner vs M74 winner), (M75 vs M76), (M77 vs M78), (M79 vs M80),
# (M81 vs M82), (M83 vs M84), (M85 vs M86), (M87 vs M88)
R16_PAIRS = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)]

# Legacy simple bracket used by pre-live simulate_tournament()
R32_BRACKET = [
    ("A","B"),("C","D"),("E","F"),("G","H"),
    ("I","J"),("K","L"),("A","C"),("B","D"),
    ("E","G"),("F","H"),("I","K"),("J","L"),
    ("A","E"),("B","F"),("C","G"),("D","H"),
]


# ---------------------------------------------------------------------------
# Probability cache — avoids recomputing the same matchup repeatedly
# ---------------------------------------------------------------------------

class ProbabilityCache:
    def __init__(self, team_features: pd.DataFrame, model=None, le=None,
                 feature_cols=None, elo=None, calibrators=None):
        self.team_features = team_features
        self.model = model
        self.le = le
        self.feature_cols = feature_cols
        self.elo = elo
        self.calibrators = calibrators
        self._cache = {}
        # In-tournament form (opponent-adjusted GD/game, z-scored). Computed once.
        try:
            from src.ingest.live_results import compute_form_z
            from src.model import WC2026_SEEDED_ELO
            self._form_z = compute_form_z(WC2026_SEEDED_ELO)
        except Exception:
            self._form_z = {}

    def get(self, home: str, away: str) -> tuple[float, float, float]:
        """Returns (p_home_win, p_draw, p_away_win)."""
        key = (home, away)
        if key not in self._cache:
            self._cache[key] = self._compute(home, away)
        return self._cache[key]

    # Host-country familiarity boost (post-prediction; WC2026 is USA/MEX/CAN)
    _MAX_HOST_BOOST = 0.04
    # In-tournament form adjustment (post-prediction; capped so 3 games can't dominate)
    _MAX_FORM_BOOST = 0.06
    _FORM_SCALE = 0.02   # per z-unit of form difference

    def _host_familiarity(self, team: str, country: str = "usa") -> float:
        col = f"host_familiarity_{country}"
        if self.team_features is None or col not in self.team_features.columns:
            return 0.0
        row = self.team_features.loc[self.team_features["team"] == team, col]
        return float(row.iloc[0]) if not row.empty else 0.0

    def _compute(self, home: str, away: str) -> tuple[float, float, float]:
        if self.model is None:
            # Fallback: equal probabilities (use before model is trained)
            return (0.40, 0.25, 0.35)

        from src.model import predict_match
        probs = predict_match(home, away, self.team_features,
                              self.model, self.le, self.feature_cols, self.elo,
                              self.calibrators)
        hw = probs.get("home_win", 1/3)
        draw = probs.get("draw", 1/3)
        aw = probs.get("away_win", 1/3)

        # Apply host familiarity boost (default USA since most WC2026 matches there)
        fam_diff = self._host_familiarity(home) - self._host_familiarity(away)
        boost = max(-self._MAX_HOST_BOOST, min(self._MAX_HOST_BOOST, fam_diff * self._MAX_HOST_BOOST))
        if boost > 0:
            transfer = min(boost, aw)
            hw, aw = hw + transfer, aw - transfer
        elif boost < 0:
            transfer = min(-boost, hw)
            hw, aw = hw - transfer, aw + transfer

        # Apply in-tournament form adjustment (live evidence the static features miss)
        form_diff = self._form_z.get(home, 0.0) - self._form_z.get(away, 0.0)
        fshift = max(-self._MAX_FORM_BOOST,
                     min(self._MAX_FORM_BOOST, form_diff * self._FORM_SCALE))
        if fshift > 0:
            transfer = min(fshift, aw)
            hw, aw = hw + transfer, aw - transfer
        elif fshift < 0:
            transfer = min(-fshift, hw)
            hw, aw = hw - transfer, aw + transfer

        total = hw + draw + aw
        if total > 0:
            hw, draw, aw = hw / total, draw / total, aw / total

        return (hw, draw, aw)


# ---------------------------------------------------------------------------
# Vectorised simulation core
# ---------------------------------------------------------------------------

def _sample_outcomes_vectorised(p_home: np.ndarray, p_draw: np.ndarray,
                                  p_away: np.ndarray,
                                  n_sims: int, xp=np) -> np.ndarray:
    """
    Sample match outcomes for N simulations in one vectorised call.

    p_home, p_draw, p_away: arrays of shape (n_matches,)
    Returns: outcome array of shape (n_sims, n_matches)
             0 = home win, 1 = draw, 2 = away win
    """
    r = xp.random.random((n_sims, len(p_home)))
    outcomes = xp.where(r < p_home, 0,
               xp.where(r < p_home + p_draw, 1, 2))
    return outcomes


def _simulate_group_stage(groups: dict, prob_cache: ProbabilityCache,
                           n_sims: int, xp=np) -> dict:
    """
    Simulate all group stage matches for all simulations at once.
    Returns dict: team → array of shape (n_sims,) with points earned.
    """
    team_points = {team: xp.zeros(n_sims, dtype=xp.int32)
                   for teams in groups.values() for team in teams}
    team_gd = {team: xp.zeros(n_sims, dtype=xp.float32)
               for teams in groups.values() for team in teams}

    for group, teams in groups.items():
        # Generate all round-robin matchups
        matchups = [(teams[i], teams[j])
                    for i in range(len(teams))
                    for j in range(i + 1, len(teams))]

        p_h = np.array([prob_cache.get(h, a)[0] for h, a in matchups])
        p_d = np.array([prob_cache.get(h, a)[1] for h, a in matchups])
        p_a = np.array([prob_cache.get(h, a)[2] for h, a in matchups])

        if xp.__name__ == "cupy":
            p_h, p_d, p_a = xp.array(p_h), xp.array(p_d), xp.array(p_a)

        outcomes = _sample_outcomes_vectorised(p_h, p_d, p_a, n_sims, xp)
        # outcomes shape: (n_sims, n_matches)

        for m_idx, (home, away) in enumerate(matchups):
            col = outcomes[:, m_idx]
            # Home win
            hw = (col == 0)
            team_points[home] += xp.where(hw, 3, 0).astype(xp.int32)
            team_points[away] += xp.where(col == 2, 3, 0).astype(xp.int32)
            # Draw
            draw = (col == 1)
            team_points[home] += xp.where(draw, 1, 0).astype(xp.int32)
            team_points[away] += xp.where(draw, 1, 0).astype(xp.int32)
            # Goal difference proxy (use point spread as tiebreaker)
            team_gd[home] += xp.where(hw, 1.0, xp.where(col == 2, -1.0, 0.0))
            team_gd[away] += xp.where(col == 2, 1.0, xp.where(hw, -1.0, 0.0))

    return team_points, team_gd


def _get_group_qualifiers(groups: dict, team_points: dict,
                           team_gd: dict, n_sims: int,
                           xp=np) -> tuple[dict, dict]:
    """
    For each simulation, determine group winners and runners-up.
    Returns (winners, runners_up): each is dict group → array(n_sims) of team indices.

    Simplified: winner = highest points in group (tiebreak: goal difference proxy).
    """
    winners = {}
    runners_up = {}

    for group, teams in groups.items():
        pts_matrix = xp.stack([team_points[t] for t in teams], axis=1)  # (n_sims, n_teams)
        gd_matrix  = xp.stack([team_gd[t]     for t in teams], axis=1)

        # Sort by points desc, then GD desc
        score = pts_matrix.astype(xp.float32) * 100 + gd_matrix
        ranking = xp.argsort(-score, axis=1)  # (n_sims, n_teams)

        winners[group]    = ranking[:, 0]   # index into `teams` list
        runners_up[group] = ranking[:, 1]

    return winners, runners_up


def _simulate_knockout_match(team_a_sims: list, team_b_sims: list,
                              all_teams: list,
                              prob_cache: ProbabilityCache,
                              n_sims: int, xp=np) -> np.ndarray:
    """
    Simulate a knockout match for each simulation.
    team_a_sims, team_b_sims: arrays of team indices (one per simulation).
    Returns: array of winning team indices (n_sims,).
    """
    # Group matchups by unique pair to use cache efficiently
    winners = np.zeros(n_sims, dtype=np.int32)

    # For knockout: no draws — if draw, use extra-time/penalty probability split
    # Adjust: home_win probability absorbs draw probability proportionally
    unique_pairs = set(zip(team_a_sims.tolist() if hasattr(team_a_sims, 'tolist')
                           else team_a_sims,
                           team_b_sims.tolist() if hasattr(team_b_sims, 'tolist')
                           else team_b_sims))

    pair_probs = {}
    for a_idx, b_idx in unique_pairs:
        a = all_teams[a_idx]
        b = all_teams[b_idx]
        ph, pd_, pa = prob_cache.get(a, b)
        # Redistribute draw probability equally (penalty shootout ~ 50/50)
        pair_probs[(a_idx, b_idx)] = (ph + pd_ / 2, pa + pd_ / 2)

    r = np.random.random(n_sims)
    for i in range(n_sims):
        a_idx = int(team_a_sims[i]) if hasattr(team_a_sims, '__getitem__') else team_a_sims
        b_idx = int(team_b_sims[i]) if hasattr(team_b_sims, '__getitem__') else team_b_sims
        p_a_wins, _ = pair_probs.get((a_idx, b_idx), (0.5, 0.5))
        winners[i] = a_idx if r[i] < p_a_wins else b_idx

    return winners


# ---------------------------------------------------------------------------
# Full tournament simulation
# ---------------------------------------------------------------------------

def simulate_tournament(n_sims: int = 100_000,
                         use_gpu: bool = False,
                         groups: dict = None,
                         team_features: pd.DataFrame = None,
                         model=None, le=None, feature_cols=None, elo=None) -> pd.DataFrame:
    """
    Run n_sims Monte Carlo simulations of WC2026.

    Returns DataFrame with columns:
        team, win_pct, final_pct, semi_pct, qf_pct, r16_pct, group_exit_pct
    """
    if groups is None:
        groups = WC2026_GROUPS

    all_teams = [t for teams in groups.values() for t in teams]
    team_idx = {t: i for i, t in enumerate(all_teams)}

    # GPU setup
    if use_gpu:
        try:
            import cupy as xp
            print(f"GPU mode: CuPy {xp.__version__}")
        except ImportError:
            print("[WARN] CuPy not installed — falling back to CPU numpy")
            import numpy as xp
            use_gpu = False
    else:
        import numpy as xp

    xp.random.seed(CFG["simulation"]["random_seed"])

    # Load team features if not provided
    if team_features is None:
        tf_path = PROCESSED_DIR / "team_features.parquet"
        team_features = pd.read_parquet(tf_path) if tf_path.exists() else pd.DataFrame()

    prob_cache = ProbabilityCache(team_features, model, le, feature_cols, elo)

    print(f"Simulating {n_sims:,} tournaments "
          f"({'GPU' if use_gpu else 'CPU'})...")

    # Stage trackers
    reached = {stage: defaultdict(int)
               for stage in ["r32", "r16", "qf", "sf", "final", "winner"]}

    # --------------- Group stage ---------------
    print("  Group stage...")
    team_points, team_gd = _simulate_group_stage(groups, prob_cache, n_sims, xp)
    winners_idx, runners_idx = _get_group_qualifiers(groups, team_points, team_gd, n_sims, xp)

    group_list = list(groups.keys())
    group_teams = list(groups.values())

    # Convert to numpy for knockout stage (easier indexing)
    def to_np(arr):
        return arr.get() if hasattr(arr, 'get') else np.array(arr)

    win_np  = {g: to_np(winners_idx[g])  for g in group_list}
    run_np  = {g: to_np(runners_idx[g])  for g in group_list}

    # Map group rank indices back to global team indices
    def resolve(g, rank_arr):
        teams = group_teams[group_list.index(g)]
        return np.array([team_idx[teams[r]] for r in rank_arr])

    win_global  = {g: resolve(g, win_np[g])  for g in group_list}
    run_global  = {g: resolve(g, run_np[g])  for g in group_list}

    # All qualifiers reached R32
    for g in group_list:
        for sim_i in range(n_sims):
            reached["r32"][all_teams[win_global[g][sim_i]]] += 1
            reached["r32"][all_teams[run_global[g][sim_i]]] += 1

    # --------------- Knockout rounds ---------------
    def run_knockout_round(stage_name, matchups_teams_a, matchups_teams_b,
                           next_stage):
        """Run one knockout round and return winners for next round."""
        round_winners = []
        for a_sims, b_sims in zip(matchups_teams_a, matchups_teams_b):
            w = _simulate_knockout_match(a_sims, b_sims, all_teams,
                                          prob_cache, n_sims)
            round_winners.append(w)
            for sim_i in range(n_sims):
                reached[next_stage][all_teams[w[sim_i]]] += 1
        return round_winners

    print("  Round of 32...")
    r32_a = [win_global[g1] for g1, g2 in R32_BRACKET]
    r32_b = [run_global[g2] for g1, g2 in R32_BRACKET]
    r16_qualifiers = run_knockout_round("r32", r32_a, r32_b, "r16")

    print("  Round of 16...")
    r16_a = r16_qualifiers[0::2]
    r16_b = r16_qualifiers[1::2]
    qf_qualifiers = run_knockout_round("r16", r16_a, r16_b, "qf")

    print("  Quarter-finals...")
    qf_a = qf_qualifiers[0::2]
    qf_b = qf_qualifiers[1::2]
    sf_qualifiers = run_knockout_round("qf", qf_a, qf_b, "sf")

    print("  Semi-finals...")
    final_qualifiers = run_knockout_round("sf",
                                           sf_qualifiers[0::2],
                                           sf_qualifiers[1::2], "final")

    print("  Final...")
    champions = run_knockout_round("final",
                                    [final_qualifiers[0]],
                                    [final_qualifiers[1]], "winner")

    # --------------- Results table ---------------
    rows = []
    for team in all_teams:
        rows.append({
            "team": team,
            "win_pct":         round(reached["winner"][team] / n_sims * 100, 2),
            "final_pct":       round(reached["final"][team]  / n_sims * 100, 2),
            "semi_pct":        round(reached["sf"][team]     / n_sims * 100, 2),
            "qf_pct":          round(reached["qf"][team]     / n_sims * 100, 2),
            "r16_pct":         round(reached["r16"][team]    / n_sims * 100, 2),
            "r32_pct":         round(reached["r32"][team]    / n_sims * 100, 2),
            "group_exit_pct":  round((1 - reached["r32"][team] / n_sims) * 100, 2),
        })

    results = pd.DataFrame(rows).sort_values("win_pct", ascending=False).reset_index(drop=True)

    # Save
    out_path = OUTPUT_DIR / f"simulation_results_{n_sims//1000}k.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")
    print(results[["team", "win_pct", "final_pct", "semi_pct"]].head(16).to_string(index=False))

    return results


# ---------------------------------------------------------------------------
# Live tournament simulation — forward-looking from current state
# ---------------------------------------------------------------------------

def _compute_group_standings_np(
    groups: dict,
    completed: list[dict],
    sim_remaining: np.ndarray,       # shape (n_remaining, n_sims): 0=home win,1=draw,2=away win
    remaining: list[dict],
    n_sims: int,
) -> tuple[dict, dict, dict]:
    """
    Compute per-simulation group standings.

    Returns:
        pts[team]  → np.ndarray (n_sims,) int32
        gd[team]   → np.ndarray (n_sims,) float32
        gf[team]   → np.ndarray (n_sims,) float32
    """
    all_teams = [t for teams in groups.values() for t in teams]
    pts = {t: np.zeros(n_sims, dtype=np.int32)   for t in all_teams}
    gd  = {t: np.zeros(n_sims, dtype=np.float32) for t in all_teams}
    gf  = {t: np.zeros(n_sims, dtype=np.float32) for t in all_teams}

    # Fixed contribution from completed matches (same for every sim)
    for m in completed:
        h, a, hg, ag = m["home"], m["away"], m["hg"], m["ag"]
        if h not in pts or a not in pts:
            continue
        gf[h] += hg; gd[h] += hg - ag
        gf[a] += ag; gd[a] += ag - hg
        if hg > ag:
            pts[h] += 3
        elif hg < ag:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1

    # Simulated remaining matches (vary per sim)
    for m_idx, m in enumerate(remaining):
        h, a = m["home"], m["away"]
        if h not in pts or a not in pts:
            continue
        outcomes = sim_remaining[m_idx]           # (n_sims,)
        hw = (outcomes == 0)
        dr = (outcomes == 1)
        aw = (outcomes == 2)
        pts[h] += np.where(hw, 3, np.where(dr, 1, 0)).astype(np.int32)
        pts[a] += np.where(aw, 3, np.where(dr, 1, 0)).astype(np.int32)
        # Use ±1 goal-difference proxy for tiebreaking
        gd[h] += np.where(hw, 1.0, np.where(aw, -1.0, 0.0))
        gd[a] += np.where(aw, 1.0, np.where(hw, -1.0, 0.0))
        gf[h] += np.where(hw, 1.0, np.where(dr, 0.5, 0.0))
        gf[a] += np.where(aw, 1.0, np.where(dr, 0.5, 0.0))

    return pts, gd, gf


def _rank_group(teams: list[str], pts: dict, gd: dict, gf: dict,
                n_sims: int) -> np.ndarray:
    """
    Rank teams within a group per simulation.
    Returns ranking array of shape (n_sims, n_teams) — indices into `teams`.
    """
    pts_m = np.stack([pts[t] for t in teams], axis=1).astype(np.float32)
    gd_m  = np.stack([gd[t]  for t in teams], axis=1)
    gf_m  = np.stack([gf[t]  for t in teams], axis=1)
    score = pts_m * 10000 + gd_m * 100 + gf_m
    return np.argsort(-score, axis=1)   # (n_sims, n_teams)


def _assign_thirds_to_slots(
    qualifying_thirds: list[tuple[str, str, float]],  # [(team, group, score)]
    third_slots: list[tuple[int, set]],               # [(r32_match_idx, eligible_groups)]
) -> dict[int, str]:
    """
    Greedily assign 8 qualifying 3rd-place teams to 8 R32 slots.
    Slot with fewest eligible groups is filled first.
    Returns: {r32_match_idx: team_name}
    """
    assignment: dict[int, str] = {}
    remaining_thirds = list(qualifying_thirds)  # sorted best→worst

    # Fill hardest slots first (fewest eligible groups)
    slots_sorted = sorted(third_slots, key=lambda x: len(x[1]))

    for slot_idx, eligible_groups in slots_sorted:
        for rank_i, (team, group, score) in enumerate(remaining_thirds):
            if group in eligible_groups:
                assignment[slot_idx] = team
                remaining_thirds.pop(rank_i)
                break
        else:
            # No eligible team left → use best remaining regardless
            if remaining_thirds:
                assignment[slot_idx] = remaining_thirds.pop(0)[0]

    return assignment


def simulate_live_tournament(
    n_sims: int = 500_000,
    use_gpu: bool = False,
    team_features: pd.DataFrame = None,
    model=None, le=None, feature_cols=None, elo=None, calibrators=None,
) -> pd.DataFrame:
    """
    Simulate WC2026 forward from current live state.

    Completed match results are fixed.
    Remaining group-stage games are simulated stochastically.
    Then knockout rounds (R32 → R16 → QF → SF → Final) are simulated
    using the official bracket structure.
    """
    from src.ingest.live_results import (
        COMPLETED_MATCHES, REMAINING_MATCHES, WC2026_ACTUAL_GROUPS,
        get_group_rank, compute_standings,
    )

    groups = WC2026_ACTUAL_GROUPS
    all_teams = [t for teams in groups.values() for t in teams]
    team_idx = {t: i for i, t in enumerate(all_teams)}
    n_teams = len(all_teams)

    # GPU / numpy
    if use_gpu:
        try:
            import cupy as xp
            print(f"GPU mode: CuPy {xp.__version__}")
        except ImportError:
            print("[WARN] CuPy not available — falling back to CPU")
            import numpy as xp
            use_gpu = False
    else:
        import numpy as xp

    np.random.seed(CFG["simulation"]["random_seed"])

    if team_features is None:
        tf_path = PROCESSED_DIR / "team_features.parquet"
        team_features = pd.read_parquet(tf_path) if tf_path.exists() else pd.DataFrame()

    prob_cache = ProbabilityCache(team_features, model, le, feature_cols, elo, calibrators)

    print(f"Simulating {n_sims:,} tournaments from live state "
          f"({'GPU' if use_gpu else 'CPU'})...")

    # ── Step 1: Simulate remaining group matches ──────────────────────────
    n_remaining = len(REMAINING_MATCHES)
    p_h = np.array([prob_cache.get(m["home"], m["away"])[0] for m in REMAINING_MATCHES])
    p_d = np.array([prob_cache.get(m["home"], m["away"])[1] for m in REMAINING_MATCHES])
    p_a = np.array([prob_cache.get(m["home"], m["away"])[2] for m in REMAINING_MATCHES])

    r = np.random.random((n_remaining, n_sims))
    sim_outcomes = np.where(r < p_h[:, None], 0,
                   np.where(r < (p_h + p_d)[:, None], 1, 2))  # (n_remaining, n_sims)

    # ── Step 2: Compute per-sim group standings ───────────────────────────
    pts, gd, gf = _compute_group_standings_np(
        groups, COMPLETED_MATCHES, sim_outcomes, REMAINING_MATCHES, n_sims
    )

    # ── Step 3: Rank teams within each group ─────────────────────────────
    # rankings[group] → shape (n_sims, 4): team indices within group list
    rankings = {}
    for g, teams in groups.items():
        rankings[g] = _rank_group(teams, pts, gd, gf, n_sims)

    def group_team_at_rank(g: str, rank: int) -> np.ndarray:
        """Return array (n_sims,) of global team indices for teams at `rank` in group g."""
        local_idx = rankings[g][:, rank]          # (n_sims,) — index within group list
        return np.array([team_idx[groups[g][i]] for i in local_idx])

    winners  = {g: group_team_at_rank(g, 0) for g in groups}
    runners  = {g: group_team_at_rank(g, 1) for g in groups}
    thirds   = {g: group_team_at_rank(g, 2) for g in groups}

    # ── Step 4: Select 8 best 3rd-place teams per simulation ─────────────
    # Score each 3rd-place team: pts*10000 + gd*100 + gf
    third_scores = np.stack(
        [pts[all_teams[thirds[g][0]]]  # placeholder; we compute per-sim below
         for g in groups], axis=1
    )
    # Rebuild properly: for each sim, get the 3rd-place team's pts/gd/gf
    group_list = list(groups.keys())
    third_score_mat = np.zeros((n_sims, 12), dtype=np.float32)
    third_idx_mat   = np.zeros((n_sims, 12), dtype=np.int32)

    for gi, g in enumerate(group_list):
        t3_global = thirds[g]                     # (n_sims,) global team idx
        for sim_i in range(n_sims):
            t = all_teams[t3_global[sim_i]]
            third_score_mat[sim_i, gi] = (pts[t][sim_i] * 10000 +
                                           gd[t][sim_i]  * 100 +
                                           gf[t][sim_i])
            third_idx_mat[sim_i, gi] = t3_global[sim_i]

    # Indices of top-8 groups by 3rd-place score, per simulation
    top8_gi = np.argsort(-third_score_mat, axis=1)[:, :8]  # (n_sims, 8)

    # ── Step 5: 3rd-place slot definitions from R32_MATCHES ──────────────
    third_slot_info = []   # [(r32_match_idx, set_of_eligible_groups)]
    for r32_i, (sa, sb) in enumerate(R32_MATCHES):
        for slot, s in [(r32_i*2, sa), (r32_i*2+1, sb)]:
            if s[0] == "T":
                eligible = set(s[1].split("/"))
                third_slot_info.append((r32_i, s, eligible))

    # ── Step 6: Build R32 matchups per simulation ─────────────────────────
    # For each R32 match, we need (team_a_idx, team_b_idx) for each sim
    r32_team_a = np.zeros((16, n_sims), dtype=np.int32)
    r32_team_b = np.zeros((16, n_sims), dtype=np.int32)

    # Pre-compute slot teams for W/R slots (deterministic given sim)
    for r32_i, (sa, sb) in enumerate(R32_MATCHES):
        for side, s, arr in [(0, sa, r32_team_a), (1, sb, r32_team_b)]:
            if s[0] == "W":
                arr[r32_i] = winners[s[1]]
            elif s[0] == "R":
                arr[r32_i] = runners[s[1]]
            # T slots handled per-sim below

    # For 3rd-place slots: assign per sim
    # Collect all T-slot info: (r32_i, side, eligible_groups)
    t_slots = []
    for r32_i, (sa, sb) in enumerate(R32_MATCHES):
        for side, s in [(0, sa), (1, sb)]:
            if s[0] == "T":
                eligible = set(s[1].split("/"))
                t_slots.append((r32_i, side, eligible))

    # T-slot arrays: use -1 as sentinel for unfilled slots
    t_slot_a = -1 * np.ones((16, n_sims), dtype=np.int32)
    t_slot_b = -1 * np.ones((16, n_sims), dtype=np.int32)

    # Pre-sort T-slots by fewest eligible options (constant across sims)
    t_slots_sorted = sorted(t_slots, key=lambda x: len(x[2]))

    # For efficiency, batch-assign 3rd-place teams
    # For each sim, map group_letter → 3rd_team_global_idx for the qualifying 8
    for sim_i in range(n_sims):
        top8 = top8_gi[sim_i]  # indices into group_list for qualifying 8
        qualifying = {}         # group_letter → global_team_idx
        for gi in top8:
            g = group_list[gi]
            qualifying[g] = third_idx_mat[sim_i, gi]

        used_groups: set[str] = set()
        unassigned: list[tuple] = []   # T-slots that couldn't be filled

        # First pass: assign from eligible qualifying groups
        for r32_i, side, eligible in t_slots_sorted:
            assigned = False
            for g in sorted(eligible, key=lambda g: -third_score_mat[sim_i, group_list.index(g)]):
                if g in qualifying and g not in used_groups:
                    t_idx = qualifying[g]
                    if side == 0:
                        t_slot_a[r32_i, sim_i] = t_idx
                    else:
                        t_slot_b[r32_i, sim_i] = t_idx
                    used_groups.add(g)
                    assigned = True
                    break
            if not assigned:
                unassigned.append((r32_i, side))

        # Fallback: fill any remaining unassigned slots with best unused qualifying team
        if unassigned:
            leftover = [(qualifying[g], third_score_mat[sim_i, group_list.index(g)])
                        for g in qualifying if g not in used_groups]
            leftover.sort(key=lambda x: -x[1])  # best first
            for (r32_i, side), (t_idx, _) in zip(unassigned, leftover):
                if side == 0:
                    t_slot_a[r32_i, sim_i] = t_idx
                else:
                    t_slot_b[r32_i, sim_i] = t_idx

    # Merge T-slots into r32_team arrays (only overwrite where T-slot was assigned)
    for r32_i, (sa, sb) in enumerate(R32_MATCHES):
        if sa[0] == "T":
            mask = t_slot_a[r32_i] >= 0
            r32_team_a[r32_i] = np.where(mask, t_slot_a[r32_i], r32_team_a[r32_i])
        if sb[0] == "T":
            mask = t_slot_b[r32_i] >= 0
            r32_team_b[r32_i] = np.where(mask, t_slot_b[r32_i], r32_team_b[r32_i])

    # ── Step 7: Count R32 qualifiers ─────────────────────────────────────
    reached = {stage: defaultdict(int)
               for stage in ["r32","r16","qf","sf","final","winner"]}

    # Teams that reached R32 — vectorized count
    for r32_i in range(16):
        for team_idx, count in zip(*np.unique(r32_team_a[r32_i], return_counts=True)):
            reached["r32"][all_teams[team_idx]] += int(count)
        for team_idx, count in zip(*np.unique(r32_team_b[r32_i], return_counts=True)):
            reached["r32"][all_teams[team_idx]] += int(count)

    # ── Step 8: Simulate knockout rounds ─────────────────────────────────
    def ko_round(a_mat: np.ndarray, b_mat: np.ndarray,
                 stage: str) -> np.ndarray:
        """
        a_mat, b_mat: shape (n_matches, n_sims)
        Returns winners: shape (n_matches, n_sims)
        """
        n_m = a_mat.shape[0]
        w = np.zeros((n_m, n_sims), dtype=np.int32)
        for m in range(n_m):
            unique_pairs = set(zip(a_mat[m].tolist(), b_mat[m].tolist()))
            pair_probs = {}
            for ai, bi in unique_pairs:
                ta, tb = all_teams[ai], all_teams[bi]
                ph, pd_, pa = prob_cache.get(ta, tb)
                pair_probs[(ai, bi)] = (ph + pd_ / 2, pa + pd_ / 2)

            # Build per-sim p_a_wins array vectorized
            p_a_arr = np.array([pair_probs.get(
                (int(a_mat[m, s]), int(b_mat[m, s])), (0.5, 0.5))[0]
                for s in range(n_sims)], dtype=np.float32)
            r = np.random.random(n_sims)
            winners_m = np.where(r < p_a_arr, a_mat[m], b_mat[m])
            w[m] = winners_m
            for team_idx, count in zip(*np.unique(winners_m, return_counts=True)):
                reached[stage][all_teams[team_idx]] += int(count)
        return w

    print("  Round of 32...")
    r16_w = ko_round(r32_team_a, r32_team_b, "r16")

    print("  Round of 16...")
    r16_a_mat = np.stack([r16_w[R16_PAIRS[i][0]] for i in range(8)])
    r16_b_mat = np.stack([r16_w[R16_PAIRS[i][1]] for i in range(8)])
    qf_w = ko_round(r16_a_mat, r16_b_mat, "qf")

    print("  Quarter-finals...")
    qf_a_mat = qf_w[0::2]
    qf_b_mat = qf_w[1::2]
    sf_w = ko_round(qf_a_mat, qf_b_mat, "sf")

    print("  Semi-finals...")
    final_w = ko_round(sf_w[0::2], sf_w[1::2], "final")

    print("  Final...")
    ko_round(final_w[0:1], final_w[1:2], "winner")

    # ── Step 9: Build results table ───────────────────────────────────────
    rows = []
    for team in all_teams:
        # Mark teams that are already eliminated (0 pts possible after all group games)
        # They still appear with 0% in all stages
        rows.append({
            "team":          team,
            "win_pct":       round(reached["winner"][team] / n_sims * 100, 2),
            "final_pct":     round(reached["final"][team]  / n_sims * 100, 2),
            "semi_pct":      round(reached["sf"][team]     / n_sims * 100, 2),
            "qf_pct":        round(reached["qf"][team]     / n_sims * 100, 2),
            "r16_pct":       round(reached["r16"][team]    / n_sims * 100, 2),
            "r32_pct":       round(reached["r32"][team]    / n_sims * 100, 2),
            "group_exit_pct":round((1 - reached["r32"][team]/n_sims)*100, 2),
        })

    results = (pd.DataFrame(rows)
               .sort_values("win_pct", ascending=False)
               .reset_index(drop=True))

    out_path = OUTPUT_DIR / f"live_simulation_{n_sims//1000}k.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")
    print(results[["team","win_pct","final_pct","semi_pct"]].head(20).to_string(index=False))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=CFG["simulation"]["n_simulations"],
                        help="Number of simulations")
    parser.add_argument("--gpu", action="store_true",
                        help="Use CuPy GPU acceleration (requires: pip install cupy-cuda12x)")
    parser.add_argument("--live", action="store_true",
                        help="Simulate forward from current live WC2026 state")
    args = parser.parse_args()

    # Load trained model + team features before simulation
    from src.model import load_model
    import pandas as pd
    _model, _le, _feature_cols, _elo, _calibrators = load_model()
    _tf_path = PROCESSED_DIR / "team_features.parquet"
    _team_features = pd.read_parquet(_tf_path) if _tf_path.exists() else pd.DataFrame()

    if args.live:
        simulate_live_tournament(n_sims=args.n, use_gpu=args.gpu,
                                  team_features=_team_features,
                                  model=_model, le=_le,
                                  feature_cols=_feature_cols, elo=_elo,
                                  calibrators=_calibrators)
    else:
        simulate_tournament(n_sims=args.n, use_gpu=args.gpu,
                             team_features=_team_features,
                             model=_model, le=_le,
                             feature_cols=_feature_cols, elo=_elo)
