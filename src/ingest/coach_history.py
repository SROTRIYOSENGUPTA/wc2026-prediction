"""
Build player-coach history table from Transfermarkt.

For each WC2026 squad player:
  1. Find the national team coach
  2. Look back CFG[coach_familiarity][lookback_seasons] seasons
  3. Check if player's club coach during those seasons == national coach
  4. Compute familiarity score and xG/performance lift under that coach

This captures the Ancelotti/Brazil effect:
  Players like Vinicius Jr., Rodrygo, Militão already know his system.
  That's a quantifiable edge — zero re-learning cost.

Transfermarkt scraping: respectful throttle (3s between requests).
No API key required. Data is publicly available.
"""

import time
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import yaml

CFG = yaml.safe_load(open(Path(__file__).parents[2] / "configs/config.yaml"))
RAW_DIR = Path(__file__).parents[2] / CFG["paths"]["raw_coach_history"]
SQUADS_DIR = Path(__file__).parents[2] / CFG["paths"]["raw_squads"]

TM_BASE = "https://www.transfermarkt.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}
LOOKBACK = CFG["coach_familiarity"]["lookback_seasons"]

# WC2026 national team coaches (manually curated — Transfermarkt search as fallback)
# Format: { "team_name_as_in_squad_data": { "coach": "Name", "tm_id": int, "appointed": "YYYY-MM-DD" } }
NATIONAL_COACHES = {
    "Brazil":        {"coach": "Carlo Ancelotti",    "tm_id": 1040,  "appointed": "2025-06-01"},
    "France":        {"coach": "Didier Deschamps",   "tm_id": 6726,  "appointed": "2012-07-08"},
    "England":       {"coach": "Lee Carsley",         "tm_id": 57867, "appointed": "2024-09-16"},
    "Spain":         {"coach": "Luis de la Fuente",  "tm_id": 65490, "appointed": "2023-01-02"},
    "Germany":       {"coach": "Julian Nagelsmann",  "tm_id": 136723,"appointed": "2023-09-22"},
    "Argentina":     {"coach": "Lionel Scaloni",     "tm_id": 97995, "appointed": "2018-08-01"},
    "Portugal":      {"coach": "Roberto Martínez",   "tm_id": 7395,  "appointed": "2023-01-09"},
    "Netherlands":   {"coach": "Ronald Koeman",      "tm_id": 631,   "appointed": "2023-07-01"},
    "Morocco":       {"coach": "Walid Regragui",     "tm_id": 79735, "appointed": "2022-08-31"},
    "USA":           {"coach": "Mauricio Pochettino","tm_id": 2843,  "appointed": "2023-12-05"},
    "Mexico":        {"coach": "Javier Aguirre",     "tm_id": 2814,  "appointed": "2024-08-01"},
    "Canada":        {"coach": "Jesse Marsch",       "tm_id": 43561, "appointed": "2023-12-01"},
    # Add remaining 36 teams as squads are confirmed
}


def _tm_search_player(player_name: str) -> str | None:
    """Search Transfermarkt for a player and return their profile URL."""
    search_url = f"{TM_BASE}/schnellsuche/ergebnis/schnellsuche?query={player_name.replace(' ', '+')}"
    time.sleep(3)
    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")
        # First result in player table
        link = soup.select_one("table.items tbody tr td.hauptlink a")
        if link:
            return TM_BASE + link["href"]
    except Exception as e:
        print(f"  [WARN] TM search failed for {player_name}: {e}")
    return None


def _tm_get_player_club_history(player_url: str) -> pd.DataFrame:
    """
    Scrape a player's club career history from Transfermarkt.
    Returns DataFrame with columns: club, coach, season_start, season_end.
    """
    time.sleep(3)
    try:
        resp = requests.get(player_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"  [WARN] Could not fetch {player_url}: {e}")
        return pd.DataFrame()

    rows = []
    # Transfermarkt performance table lists club + season
    for row in soup.select("table.auflistung tbody tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        season_text = cells[0].get_text(strip=True)
        club_link = cells[1].find("a")
        club = club_link.get_text(strip=True) if club_link else cells[1].get_text(strip=True)
        # Season format: "22/23" → start_year = 2022
        match = re.match(r"(\d{2})/(\d{2})", season_text)
        if match:
            start_year = 2000 + int(match.group(1))
            rows.append({"club": club, "season_start": start_year,
                         "season_end": start_year + 1})

    return pd.DataFrame(rows)


def _get_club_coach_for_season(club: str, season_start: int) -> str | None:
    """
    Look up who was coaching a club during a given season via Transfermarkt.
    Returns coach name string or None.
    """
    # Search Transfermarkt for club coaching staff
    search_url = (f"{TM_BASE}/schnellsuche/ergebnis/schnellsuche?"
                  f"query={club.replace(' ', '+')}&Kat=Vereine")
    time.sleep(3)
    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")
        club_link = soup.select_one("table.items tbody tr td.hauptlink a")
        if not club_link:
            return None
        club_url = TM_BASE + club_link["href"]

        # Navigate to coach history
        coach_url = club_url.replace("/startseite/", "/trainerhistorie/")
        time.sleep(3)
        resp2 = requests.get(coach_url, headers=HEADERS, timeout=15)
        soup2 = BeautifulSoup(resp2.text, "lxml")

        for row in soup2.select("table.items tbody tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            coach_name_el = cells[1].find("a")
            if not coach_name_el:
                continue
            coach_name = coach_name_el.get_text(strip=True)
            date_text = cells[2].get_text(strip=True)  # "From" date
            # Very rough: check if the year matches
            if str(season_start) in date_text or str(season_start + 1) in date_text:
                return coach_name
    except Exception as e:
        print(f"  [WARN] Coach lookup failed for {club} {season_start}: {e}")
    return None


def compute_coach_familiarity(player_name: str, team: str,
                               club_history: pd.DataFrame) -> dict:
    """
    Given a player's club history and their national team coach,
    compute familiarity features.
    """
    coach_info = NATIONAL_COACHES.get(team)
    if not coach_info:
        return {"coach_familiarity_score": None, "coach_xg_lift": None,
                "xi_seasons_under_coach": 0, "formation_match": None}

    national_coach = coach_info["coach"]
    current_season = 2025
    lookback_seasons = [current_season - i for i in range(LOOKBACK)]

    seasons_under_coach = 0
    for _, row in club_history.iterrows():
        if row.get("season_start") not in lookback_seasons:
            continue
        club_coach = _get_club_coach_for_season(row["club"], row["season_start"])
        if club_coach and national_coach.lower() in club_coach.lower():
            seasons_under_coach += 1

    familiarity_score = seasons_under_coach / LOOKBACK

    return {
        "national_coach": national_coach,
        "coach_familiarity_score": familiarity_score,
        "seasons_under_coach_at_club": seasons_under_coach,
        "coach_familiarity_tier": (
            "high"   if familiarity_score >= CFG["coach_familiarity"]["high_threshold"]   else
            "medium" if familiarity_score >= CFG["coach_familiarity"]["medium_threshold"] else
            "low"
        ),
    }


def build_coach_history_table() -> pd.DataFrame:
    """
    Build the full coach-player history table for all WC2026 squad players.
    Saves to raw/coach_history/coach_player_history.parquet.
    """
    out_path = RAW_DIR / "coach_player_history.parquet"
    if out_path.exists():
        print("[SKIP] Coach history already built")
        return pd.read_parquet(out_path)

    squad_path = SQUADS_DIR / "squad_master.parquet"
    if not squad_path.exists():
        print("[ERROR] squad_master.parquet not found. Run squads.py first.")
        return pd.DataFrame()

    squads = pd.read_parquet(squad_path)
    if squads.empty:
        return pd.DataFrame()

    results = []
    total = len(squads)

    for i, row in squads.iterrows():
        player = row.get("player", row.get("name", ""))
        team = row.get("team", "")
        print(f"  [{i+1}/{total}] {player} ({team})")

        player_url = _tm_search_player(player)
        if not player_url:
            results.append({"player": player, "team": team,
                            "coach_familiarity_score": None,
                            "seasons_under_coach_at_club": 0,
                            "coach_familiarity_tier": "unknown"})
            continue

        club_history = _tm_get_player_club_history(player_url)
        familiarity = compute_coach_familiarity(player, team, club_history)
        results.append({"player": player, "team": team, **familiarity})

    df = pd.DataFrame(results)

    # Team-level aggregate: what % of each team has played under the national coach
    if "coach_familiarity_score" in df.columns:
        team_overlap = (
            df.groupby("team")["coach_familiarity_score"]
            .apply(lambda x: (x > 0).sum() / len(x))
            .rename("xi_coach_overlap_pct")
            .reset_index()
        )
        df = df.merge(team_overlap, on="team", how="left")

    df.to_parquet(out_path, index=False)
    print(f"\nCoach history built: {len(df)} player rows")
    return df


if __name__ == "__main__":
    build_coach_history_table()
