"""
WC2026 live match results and current standings.
Last updated: 2026-06-27 — GROUP STAGE COMPLETE (all 72 matches played).
Knockout rounds (Round of 32) begin June 28.

Source: CBS Sports / ESPN / NBC Sports / FotMob match-by-match results.
"""

from __future__ import annotations
from collections import defaultdict

# ---------------------------------------------------------------------------
# All completed match results
# ---------------------------------------------------------------------------
COMPLETED_MATCHES: list[dict] = [
    # ── Group A (COMPLETE) ──────────────────────────────────────────────────
    {"group":"A","home":"Mexico",        "away":"South Africa",  "hg":2,"ag":0,"date":"2026-06-11"},
    {"group":"A","home":"South Korea",   "away":"Czechia",       "hg":2,"ag":1,"date":"2026-06-11"},
    {"group":"A","home":"Czechia",       "away":"South Africa",  "hg":1,"ag":1,"date":"2026-06-18"},
    {"group":"A","home":"Mexico",        "away":"South Korea",   "hg":1,"ag":0,"date":"2026-06-18"},
    {"group":"A","home":"Czechia",       "away":"Mexico",        "hg":0,"ag":3,"date":"2026-06-24"},
    {"group":"A","home":"South Africa",  "away":"South Korea",   "hg":1,"ag":0,"date":"2026-06-24"},
    # ── Group B (COMPLETE) ──────────────────────────────────────────────────
    {"group":"B","home":"Bosnia and Herzegovina","away":"Canada",      "hg":1,"ag":1,"date":"2026-06-12"},
    {"group":"B","home":"Qatar",                 "away":"Switzerland", "hg":1,"ag":1,"date":"2026-06-13"},
    {"group":"B","home":"Switzerland","away":"Bosnia and Herzegovina", "hg":4,"ag":1,"date":"2026-06-18"},
    {"group":"B","home":"Canada",      "away":"Qatar",                 "hg":6,"ag":0,"date":"2026-06-18"},
    {"group":"B","home":"Switzerland", "away":"Canada",                "hg":2,"ag":1,"date":"2026-06-24"},
    {"group":"B","home":"Bosnia and Herzegovina","away":"Qatar",       "hg":3,"ag":1,"date":"2026-06-24"},
    # ── Group C (COMPLETE) ──────────────────────────────────────────────────
    {"group":"C","home":"Brazil",  "away":"Morocco",  "hg":1,"ag":1,"date":"2026-06-13"},
    {"group":"C","home":"Haiti",   "away":"Scotland", "hg":0,"ag":1,"date":"2026-06-13"},
    {"group":"C","home":"Morocco", "away":"Scotland", "hg":1,"ag":0,"date":"2026-06-19"},
    {"group":"C","home":"Brazil",  "away":"Haiti",    "hg":3,"ag":0,"date":"2026-06-19"},
    {"group":"C","home":"Scotland","away":"Brazil",   "hg":0,"ag":3,"date":"2026-06-24"},
    {"group":"C","home":"Morocco", "away":"Haiti",    "hg":4,"ag":2,"date":"2026-06-24"},
    # ── Group D (COMPLETE) ──────────────────────────────────────────────────
    {"group":"D","home":"USA",       "away":"Paraguay",  "hg":4,"ag":1,"date":"2026-06-12"},
    {"group":"D","home":"Australia", "away":"Turkey",    "hg":2,"ag":0,"date":"2026-06-13"},
    {"group":"D","home":"USA",       "away":"Australia", "hg":2,"ag":0,"date":"2026-06-19"},
    {"group":"D","home":"Paraguay",  "away":"Turkey",    "hg":1,"ag":0,"date":"2026-06-19"},
    {"group":"D","home":"Turkey",    "away":"USA",       "hg":3,"ag":2,"date":"2026-06-25"},
    {"group":"D","home":"Paraguay",  "away":"Australia", "hg":0,"ag":0,"date":"2026-06-25"},
    # ── Group E (COMPLETE) ──────────────────────────────────────────────────
    {"group":"E","home":"Germany",     "away":"Curaçao",     "hg":7,"ag":1,"date":"2026-06-14"},
    {"group":"E","home":"Ivory Coast", "away":"Ecuador",     "hg":1,"ag":0,"date":"2026-06-14"},
    {"group":"E","home":"Germany",     "away":"Ivory Coast", "hg":2,"ag":1,"date":"2026-06-20"},
    {"group":"E","home":"Ecuador",     "away":"Curaçao",     "hg":0,"ag":0,"date":"2026-06-20"},
    {"group":"E","home":"Ecuador",     "away":"Germany",     "hg":2,"ag":1,"date":"2026-06-25"},
    {"group":"E","home":"Curaçao",     "away":"Ivory Coast", "hg":0,"ag":2,"date":"2026-06-25"},
    # ── Group F (COMPLETE) ──────────────────────────────────────────────────
    {"group":"F","home":"Netherlands","away":"Japan",        "hg":2,"ag":2,"date":"2026-06-14"},
    {"group":"F","home":"Sweden",     "away":"Tunisia",      "hg":5,"ag":1,"date":"2026-06-14"},
    {"group":"F","home":"Netherlands","away":"Sweden",       "hg":5,"ag":1,"date":"2026-06-20"},
    {"group":"F","home":"Tunisia",    "away":"Japan",        "hg":0,"ag":4,"date":"2026-06-20"},
    {"group":"F","home":"Japan",      "away":"Sweden",       "hg":1,"ag":1,"date":"2026-06-25"},
    {"group":"F","home":"Tunisia",    "away":"Netherlands",  "hg":1,"ag":3,"date":"2026-06-25"},
    # ── Group G (COMPLETE) ──────────────────────────────────────────────────
    {"group":"G","home":"Belgium",     "away":"Egypt",        "hg":1,"ag":1,"date":"2026-06-15"},
    {"group":"G","home":"Iran",        "away":"New Zealand",  "hg":2,"ag":2,"date":"2026-06-15"},
    {"group":"G","home":"Belgium",     "away":"Iran",         "hg":0,"ag":0,"date":"2026-06-21"},
    {"group":"G","home":"Egypt",       "away":"New Zealand",  "hg":3,"ag":1,"date":"2026-06-21"},
    {"group":"G","home":"Egypt",       "away":"Iran",         "hg":1,"ag":1,"date":"2026-06-26"},
    {"group":"G","home":"Belgium",     "away":"New Zealand",  "hg":5,"ag":1,"date":"2026-06-26"},
    # ── Group H (COMPLETE) ──────────────────────────────────────────────────
    {"group":"H","home":"Spain",        "away":"Cabo Verde",  "hg":0,"ag":0,"date":"2026-06-15"},
    {"group":"H","home":"Saudi Arabia", "away":"Uruguay",     "hg":1,"ag":1,"date":"2026-06-15"},
    {"group":"H","home":"Spain",        "away":"Saudi Arabia","hg":4,"ag":0,"date":"2026-06-21"},
    {"group":"H","home":"Uruguay",      "away":"Cabo Verde",  "hg":2,"ag":2,"date":"2026-06-21"},
    {"group":"H","home":"Cabo Verde",   "away":"Saudi Arabia","hg":0,"ag":0,"date":"2026-06-26"},
    {"group":"H","home":"Spain",        "away":"Uruguay",     "hg":1,"ag":0,"date":"2026-06-26"},
    # ── Group I (COMPLETE) ──────────────────────────────────────────────────
    {"group":"I","home":"France",   "away":"Senegal", "hg":3,"ag":1,"date":"2026-06-16"},
    {"group":"I","home":"Iraq",     "away":"Norway",  "hg":1,"ag":4,"date":"2026-06-16"},
    {"group":"I","home":"France",   "away":"Iraq",    "hg":3,"ag":0,"date":"2026-06-22"},
    {"group":"I","home":"Norway",   "away":"Senegal", "hg":3,"ag":2,"date":"2026-06-22"},
    {"group":"I","home":"France",   "away":"Norway",  "hg":4,"ag":1,"date":"2026-06-26"},
    {"group":"I","home":"Senegal",  "away":"Iraq",    "hg":5,"ag":0,"date":"2026-06-26"},
    # ── Group J (COMPLETE) ──────────────────────────────────────────────────
    {"group":"J","home":"Argentina","away":"Algeria", "hg":3,"ag":0,"date":"2026-06-16"},
    {"group":"J","home":"Austria",  "away":"Jordan",  "hg":3,"ag":1,"date":"2026-06-16"},
    {"group":"J","home":"Argentina","away":"Austria", "hg":2,"ag":0,"date":"2026-06-22"},
    {"group":"J","home":"Algeria",  "away":"Jordan",  "hg":2,"ag":1,"date":"2026-06-22"},
    {"group":"J","home":"Jordan",   "away":"Argentina","hg":1,"ag":3,"date":"2026-06-27"},
    {"group":"J","home":"Algeria",  "away":"Austria", "hg":3,"ag":2,"date":"2026-06-27"},
    # ── Group K (COMPLETE) ──────────────────────────────────────────────────
    {"group":"K","home":"Portugal","away":"DR Congo",   "hg":1,"ag":1,"date":"2026-06-17"},
    {"group":"K","home":"Colombia","away":"Uzbekistan", "hg":3,"ag":1,"date":"2026-06-17"},
    {"group":"K","home":"Portugal","away":"Uzbekistan", "hg":5,"ag":0,"date":"2026-06-23"},
    {"group":"K","home":"Colombia","away":"DR Congo",   "hg":1,"ag":0,"date":"2026-06-23"},
    {"group":"K","home":"Colombia","away":"Portugal",   "hg":0,"ag":0,"date":"2026-06-27"},
    {"group":"K","home":"DR Congo","away":"Uzbekistan", "hg":3,"ag":1,"date":"2026-06-27"},
    # ── Group L (COMPLETE) ──────────────────────────────────────────────────
    {"group":"L","home":"England", "away":"Croatia", "hg":4,"ag":2,"date":"2026-06-17"},
    {"group":"L","home":"Ghana",   "away":"Panama",  "hg":1,"ag":0,"date":"2026-06-17"},
    {"group":"L","home":"England", "away":"Ghana",   "hg":0,"ag":0,"date":"2026-06-23"},
    {"group":"L","home":"Croatia", "away":"Panama",  "hg":1,"ag":0,"date":"2026-06-23"},
    {"group":"L","home":"Panama",  "away":"England", "hg":0,"ag":2,"date":"2026-06-27"},
    {"group":"L","home":"Croatia", "away":"Ghana",   "hg":2,"ag":1,"date":"2026-06-27"},
]

# Group stage COMPLETE as of 2026-06-27. Knockout rounds (R32) begin June 28.
REMAINING_MATCHES: list[dict] = []

# Completed KNOCKOUT results (R32 onward). These are locked into the bracket:
# the actual winner advances regardless of model probability.
# Penalty-shootout games are recorded with their 1-1 (etc.) regulation score for ELO,
# with `winner` carrying who advanced.
COMPLETED_KNOCKOUT: list[dict] = [
    {"stage":"R32","home":"South Africa","away":"Canada",   "hg":0,"ag":1,"winner":"Canada",   "date":"2026-06-28"},
    {"stage":"R32","home":"Brazil",      "away":"Japan",    "hg":2,"ag":1,"winner":"Brazil",   "date":"2026-06-29"},
    {"stage":"R32","home":"Germany",     "away":"Paraguay", "hg":1,"ag":1,"winner":"Paraguay", "date":"2026-06-29","note":"Paraguay won 4-3 on penalties"},
    {"stage":"R32","home":"Netherlands", "away":"Morocco",  "hg":1,"ag":1,"winner":"Morocco",  "date":"2026-06-29","note":"Morocco won 3-2 on penalties"},
    {"stage":"R32","home":"Ivory Coast", "away":"Norway",   "hg":1,"ag":2,"winner":"Norway",   "date":"2026-06-30"},
    {"stage":"R32","home":"France",      "away":"Sweden",   "hg":3,"ag":0,"winner":"France",   "date":"2026-06-30"},
    {"stage":"R32","home":"Mexico",      "away":"Ecuador",  "hg":2,"ag":0,"winner":"Mexico",   "date":"2026-06-30"},
    {"stage":"R32","home":"England",     "away":"DR Congo", "hg":2,"ag":1,"winner":"England",  "date":"2026-07-01"},
    {"stage":"R32","home":"Belgium",     "away":"Senegal",  "hg":3,"ag":2,"winner":"Belgium",  "date":"2026-07-01","note":"after extra time"},
    {"stage":"R32","home":"USA",         "away":"Bosnia and Herzegovina","hg":2,"ag":0,"winner":"USA","date":"2026-07-01"},
    {"stage":"R32","home":"Spain",       "away":"Austria",  "hg":3,"ag":0,"winner":"Spain",    "date":"2026-07-02"},
    {"stage":"R32","home":"Portugal",    "away":"Croatia",  "hg":2,"ag":1,"winner":"Portugal", "date":"2026-07-02"},
    {"stage":"R32","home":"Switzerland", "away":"Algeria",  "hg":2,"ag":0,"winner":"Switzerland","date":"2026-07-02"},
    {"stage":"R32","home":"Australia",   "away":"Egypt",    "hg":1,"ag":1,"winner":"Egypt",    "date":"2026-07-03","note":"Egypt won 4-2 on penalties"},
    {"stage":"R32","home":"Argentina",   "away":"Cabo Verde","hg":3,"ag":2,"winner":"Argentina","date":"2026-07-03","note":"after extra time"},
    {"stage":"R32","home":"Colombia",    "away":"Ghana",    "hg":1,"ag":0,"winner":"Colombia", "date":"2026-07-03"},
    # Round of 16 (Jul 4–7) — add each here once FINAL, verified from a reliable score.
    {"stage":"R16","home":"Paraguay",    "away":"France",   "hg":0,"ag":1,"winner":"France",   "date":"2026-07-04"},
    {"stage":"R16","home":"Canada",      "away":"Morocco",  "hg":0,"ag":3,"winner":"Morocco",  "date":"2026-07-04"},
    {"stage":"R16","home":"Portugal",    "away":"Spain",    "hg":0,"ag":1,"winner":"Spain",    "date":"2026-07-06"},
    {"stage":"R16","home":"USA",         "away":"Belgium",  "hg":1,"ag":4,"winner":"Belgium",  "date":"2026-07-06"},
    {"stage":"R16","home":"Brazil",      "away":"Norway",   "hg":1,"ag":2,"winner":"Norway",   "date":"2026-07-05","note":"Norway upset Brazil"},
    {"stage":"R16","home":"Mexico",      "away":"England",  "hg":2,"ag":3,"winner":"England",  "date":"2026-07-05","note":"England won 3-2 with 10 men"},
    {"stage":"R16","home":"Argentina",   "away":"Egypt",    "hg":3,"ag":2,"winner":"Argentina","date":"2026-07-07"},
    {"stage":"R16","home":"Switzerland", "away":"Colombia", "hg":0,"ag":0,"winner":"Switzerland","date":"2026-07-07","note":"Switzerland won on penalties"},
    # Quarter-finals (Jul 9-11)
    {"stage":"QF","home":"France",       "away":"Morocco",  "hg":2,"ag":0,"winner":"France",   "date":"2026-07-09"},
    {"stage":"QF","home":"Spain",        "away":"Belgium",  "hg":2,"ag":1,"winner":"Spain",    "date":"2026-07-10"},
    {"stage":"QF","home":"Norway",       "away":"England",  "hg":1,"ag":2,"winner":"England",  "date":"2026-07-11","note":"England into the semis"},
]

# ---------------------------------------------------------------------------
# Individual "hot-hand" — a team's best in-tournament attacker, scored as
# goals + 0.5*assists (as of the quarter-finals; FIFA/press Golden Boot tracker).
# A player on a scoring tear (Mbappé, Messi, Haaland) is live evidence the static
# squad features can't see. Consumed as a small, capped post-prediction boost.
# Teams not listed default to 0 (no standout scorer).
# ---------------------------------------------------------------------------
TOURNAMENT_TOP_ATTACKER: dict[str, float] = {
    "France":      9.0,   # Mbappé — 8 goals + 2 assists
    "Argentina":   8.0,   # Messi — 8 goals
    "Norway":      7.0,   # Haaland — 7 goals
    "England":     6.0,   # Kane — 6 goals
    "Spain":       4.5,   # Oyarzabal ~4 (+a) / Yamal
    "Switzerland": 2.0,   # Embolo / Ndoye
    # eliminated sides (kept for reference; they no longer feature in live games)
    "Brazil":      5.0,   # Vinícius Jr.
    "Belgium":     3.0,   # Lukaku / Tielemans
    "Morocco":     3.0,   # Rahimi / Diop
}

# Actual WC2026 groups as officially drawn
WC2026_ACTUAL_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Switzerland", "Canada", "Bosnia and Herzegovina", "Qatar"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["USA", "Australia", "Paraguay", "Turkey"],
    "E": ["Germany", "Ivory Coast", "Ecuador", "Curaçao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Egypt", "Iran", "Belgium", "New Zealand"],
    "H": ["Spain", "Uruguay", "Cabo Verde", "Saudi Arabia"],
    "I": ["France", "Norway", "Senegal", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Colombia", "Portugal", "DR Congo", "Uzbekistan"],
    "L": ["England", "Ghana", "Croatia", "Panama"],
}

# ---------------------------------------------------------------------------
# Compute group standings from match results
# ---------------------------------------------------------------------------

def compute_standings(matches: list[dict]) -> dict[str, dict[str, dict]]:
    """
    Compute group standings from a list of match result dicts.
    Returns: {group: {team: {pts, w, d, l, gf, ga, gd}}}
    """
    standings: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(
        lambda: {"pts":0, "w":0, "d":0, "l":0, "gf":0, "ga":0, "gd":0}
    ))

    for m in matches:
        g, h, a, hg, ag = m["group"], m["home"], m["away"], m["hg"], m["ag"]
        sh, sa = standings[g][h], standings[g][a]
        sh["gf"] += hg; sh["ga"] += ag; sh["gd"] += hg - ag
        sa["gf"] += ag; sa["ga"] += hg; sa["gd"] += ag - hg
        if hg > ag:
            sh["pts"] += 3; sh["w"] += 1; sa["l"] += 1
        elif hg < ag:
            sa["pts"] += 3; sa["w"] += 1; sh["l"] += 1
        else:
            sh["pts"] += 1; sh["d"] += 1
            sa["pts"] += 1; sa["d"] += 1

    return {g: dict(teams) for g, teams in standings.items()}


def get_current_standings() -> dict[str, dict[str, dict]]:
    """Return standings from completed matches only."""
    return compute_standings(COMPLETED_MATCHES)


def get_group_rank(standings: dict[str, dict], group_teams: list[str]) -> list[str]:
    """
    Sort a group's teams by: pts desc → gd desc → gf desc → name asc.
    Returns ranked list of team names.
    """
    def key(t):
        s = standings.get(t, {"pts":0,"gd":0,"gf":0})
        return (-s["pts"], -s["gd"], -s["gf"], t)
    return sorted(group_teams, key=key)


def compute_form_z(seeded_elo: dict, matches: list[dict] | None = None) -> dict[str, float]:
    """In-tournament FORM signal — opponent-adjusted goal difference per game, z-scored.

    Static squad features (club xG, Ballon d'Or, chemistry) describe a team's
    *potential*; they can't see that a star's club form isn't translating, or that
    a scorer is hot right now. This captures the live signal: each WC2026 result's
    goal difference is weighted by the opponent's pre-tournament ELO (beating a
    strong side counts more), averaged per game, then standardized across the field.

    Positive z = over-performing the field so far; negative = under-performing.
    Used as a small, capped post-prediction adjustment in the bracket + simulation.
    """
    import statistics
    matches = COMPLETED_MATCHES if matches is None else matches
    teams = sorted({t for g in WC2026_ACTUAL_GROUPS.values() for t in g})
    mean_elo = sum(seeded_elo.values()) / max(len(seeded_elo), 1)
    wperf = {t: 0.0 for t in teams}
    n = {t: 0 for t in teams}
    for m in matches:
        h, a, hg, ag = m["home"], m["away"], m["hg"], m["ag"]
        if h not in wperf or a not in wperf:
            continue
        wperf[h] += (hg - ag) * (seeded_elo.get(a, 1500) / mean_elo); n[h] += 1
        wperf[a] += (ag - hg) * (seeded_elo.get(h, 1500) / mean_elo); n[a] += 1
    perf = {t: wperf[t] / n[t] for t in teams if n[t] > 0}
    if len(perf) < 2:
        return {t: 0.0 for t in perf}
    mu = statistics.mean(perf.values())
    sd = statistics.pstdev(perf.values()) or 1.0
    return {t: (perf[t] - mu) / sd for t in perf}
