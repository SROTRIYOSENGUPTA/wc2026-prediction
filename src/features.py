"""
Feature engineering: merges all four data streams into model-ready feature vectors.

Four streams:
  1. Club season data     → rolling form, per-90 stats, league difficulty
  2. International data   → tournament-weighted form, caps, xG at intl level
  3. Squad data           → WC2026 roster baseline
  4. Coach history        → familiarity score, xi_coach_overlap_pct

Key output features per player:
  - club_xg_p90, club_xa_p90, club_npxg_p90 (club form)
  - intl_xg_p90, intl_xa_p90 (international form, tournament-weighted)
  - club_to_intl_xg_delta  ← the Raphinha signal
  - coach_familiarity_score, coach_familiarity_tier
  - rolling_form_weighted (last 5 intl matches, time-decayed)
  - avg_caps, cohesion_proxy

Key output features per team:
  - team_net_xg (rolling weighted)
  - team_ppda
  - team_xi_coach_overlap_pct
  - squad_rating_attack/midfield/defense
  - avg_age, avg_caps
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

CFG = yaml.safe_load(open(Path(__file__).parents[1] / "configs/config.yaml"))
PROCESSED_DIR = Path(__file__).parents[1] / CFG["paths"]["processed"]
RAW_CLUB = Path(__file__).parents[1] / CFG["paths"]["raw_club_seasons"]
RAW_INTL = Path(__file__).parents[1] / CFG["paths"]["raw_international"]
RAW_SQUADS = Path(__file__).parents[1] / CFG["paths"]["raw_squads"]
RAW_COACH = Path(__file__).parents[1] / CFG["paths"]["raw_coach_history"]

FORM_WEIGHTS = list(CFG["form_weights"].values())  # [0.40, 0.25, 0.15, 0.12, 0.08]
MIN_CAPS = CFG["min_intl_caps_for_form"]


# ---------------------------------------------------------------------------
# Canonical WC2026 display names — maps StatsBomb/FIFA names → WC2026 display names
# Applied to the "team" column in player_features and team_features so the model
# and simulation always use the same names as live_results.py / WC2026_ACTUAL_GROUPS.
# ---------------------------------------------------------------------------
TEAM_NAME_NORMALIZE: dict[str, str] = {
    "United States":          "USA",
    "Türkiye":                "Turkey",
    "Czech Republic":         "Czechia",
    "Korea Republic":         "South Korea",
    "Côte d'Ivoire":          "Ivory Coast",
    "Cape Verde Islands":     "Cabo Verde",
    "Congo DR":               "DR Congo",
    "Curacao":                "Curaçao",
    # StatsBomb sometimes uses these alternate spellings
    "DR Congo":               "DR Congo",         # already canonical
    "Ivory Coast":            "Ivory Coast",
    "Bosnia-Herzegovina":     "Bosnia and Herzegovina",
    "Bosnia & Herzegovina":   "Bosnia and Herzegovina",
    "Cote d'Ivoire":          "Ivory Coast",
    "Türkiye":                "Turkey",
}


# ---------------------------------------------------------------------------
# League difficulty — Big 5 (used by FBref pull) + all others (used by RT fallback)
# Multiplier: 1.00 = Premier League level. Goals scored in weaker leagues are
# scaled DOWN so a 0.5 goals/90 in Qatar ≠ 0.5 goals/90 in the PL.
# ---------------------------------------------------------------------------
LEAGUE_DIFFICULTY = {
    # Big 5 (FBref primary source)
    "premier_league":        1.00,
    "la_liga":               0.97,
    "bundesliga":            0.95,
    "serie_a":               0.94,
    "ligue_1":               0.88,
    # European second tier
    "english_championship":  0.82,
    "eredivisie":            0.83,
    "belgian_pro_league":    0.80,
    "primeira_liga":         0.80,
    "turkish_super_lig":     0.80,
    "scottish_premiership":  0.78,
    "swiss_super_league":    0.76,
    "austrian_bundesliga":   0.74,
    "danish_superliga":      0.74,
    "norwegian_eliteserien": 0.74,
    "swedish_allsvenskan":   0.74,
    "english_league_one":    0.72,
    "greek_super_league":    0.72,
    "polish_ekstraklasa":    0.72,
    "czech_first_league":    0.72,
    "russian_premier":       0.72,
    "israeli_premier":       0.72,
    "cypriot_first":         0.65,
    "serbian_super":         0.70,
    "croatian_super":        0.68,
    "hungarian_otp":         0.65,
    "romanian_liga":         0.66,
    "slovenian_pnl":         0.65,
    "bosnian_premier":       0.62,
    "belarusian_premier":    0.62,
    "georgian_erovnuli":     0.60,
    "kazakh_premier":        0.60,
    "uzbek_super":           0.62,
    "ukrainian_premier":     0.73,
    "bulgarian_first":       0.65,
    # Americas
    "brasileirao":           0.78,
    "arg_primera":           0.76,
    "liga_mx":               0.75,
    "colombian_primera":     0.70,
    "chilean_primera":       0.68,
    "uruguayan_primera":     0.68,
    "ecuadorian_serie_a":    0.65,
    "venezuelan_primera":    0.60,
    "mls":                   0.70,
    "canadian_premier":      0.62,
    # Asia / Middle East
    "saudi_pro":             0.72,
    "uae_pro":               0.62,
    "qatar_stars":           0.60,
    "iranian_persian_gulf":  0.65,
    "iraqi_premier":         0.58,
    "jordanian_premier":     0.58,
    "j1_league":             0.73,
    "k_league":              0.72,
    "indonesian_liga":       0.58,
    "malaysian_super":       0.58,
    # Africa
    "egyptian_premier":      0.63,
    "moroccan_botola":       0.62,
    "algerian_ligue":        0.60,
    "tunisian_ligue":        0.62,
    "south_african_psl":     0.60,
    # Oceania
    "a_league":              0.68,
    # Other European
    "irish_premier":         0.62,
    "estonian_meistriliiga": 0.58,
    "faroese_premier":       0.55,
}

# ---------------------------------------------------------------------------
# Club prestige tiers — trophy culture + UCL-level competition exposure.
# Scale: 1.0 = perennial UCL/title winner, 0.0 = amateur/unknown.
# Used to compute squad_club_prestige (position-weighted, FW/MF/DF).
# ---------------------------------------------------------------------------
CLUB_PRESTIGE: dict[str, float] = {
    # ── Tier 1 (1.0) — Absolute elite: 2025-26 silverware / UCL top 4 ────
    # PSG: UCL back-to-back winner (2024-25 and 2025-26)
    "Paris Saint Germain": 1.00, "Paris Saint-Germain": 1.00,
    # Arsenal: UCL finalist + Premier League winner 2025-26 (first PL in 22 years)
    "Arsenal": 1.00,
    # Barcelona: La Liga back-to-back 2024-25 and 2025-26, UCL quarter-finalist
    "FC Barcelona": 1.00,
    # Bayern: Bundesliga winner 2025-26, UCL semi-finalist (beat Real Madrid)
    "FC Bayern München": 1.00, "FC Bayern": 1.00,
    # Atletico Madrid: UCL semi-finalist 2025-26
    "Atlético Madrid": 1.00,
    # Real Madrid: historical GOAT club — 15 UCL titles. UCL QF 2025-26.
    "Real Madrid": 1.00,
    # Inter: Serie A winner 2025-26, UCL regular
    "Inter": 0.97,
    # Liverpool: UCL quarter-finalist 2025-26 (lost 4-0 to PSG)
    "Liverpool": 0.95,
    # Man City: UCL R16 exit (lost to Real Madrid), lost PL to Arsenal — down from peak
    "Manchester City": 0.92,
    # Chelsea: 8-2 UCL exit to PSG in R16, no major trophies — significant drop
    "Chelsea": 0.82,
    # Bayer Leverkusen: lost Bundesliga back to Bayern after historic 2023-24 — step down
    "Bayer 04 Leverkusen": 0.88,
    "Borussia Dortmund": 0.93,
    "Juventus": 0.88, "AC Milan": 0.88, "Manchester United": 0.83,
    # ── Tier 2 (0.75) — Strong European: Europa/CL contenders ─────────────
    "Tottenham Hotspur": 0.78, "Newcastle United": 0.76, "Aston Villa": 0.76,
    "Brighton & Hove Albion": 0.72, "West Ham United": 0.72,
    "Wolverhampton Wanderers": 0.70, "Crystal Palace": 0.68,
    "Fulham": 0.68, "Brentford": 0.67, "Everton": 0.65,
    "AFC Bournemouth": 0.65, "Nottingham Forest": 0.65,
    "Atalanta": 0.80, "Roma": 0.75, "Napoli": 0.85, "Lazio": 0.72,
    "VfB Stuttgart": 0.74, "Eintracht Frankfurt": 0.74, "RB Leipzig": 0.78,
    "TSG Hoffenheim": 0.65, "Borussia Mönchengladbach": 0.68,
    "VfL Wolfsburg": 0.65, "FSV Mainz 05": 0.65, "Werder Bremen": 0.65,
    "FC Augsburg": 0.60, "SC Freiburg": 0.65, "FC Köln": 0.62,
    "Real Sociedad": 0.74, "Athletic Club": 0.72, "Villarreal": 0.74,
    "Real Betis": 0.70, "Sevilla": 0.76, "Celta de Vigo": 0.62,
    "Valencia": 0.65, "Mallorca": 0.62, "Osasuna": 0.60,
    "Girona": 0.65, "Rayo Vallecano": 0.58,
    "LOSC Lille": 0.72, "Olympique Marseille": 0.72, "Monaco": 0.72,
    "Olympique Lyonnais": 0.70, "Nice": 0.68, "Rennes": 0.65,
    "Strasbourg": 0.60, "Stade Brest": 0.62,
    "Feyenoord": 0.74, "PSV": 0.78, "Ajax": 0.80,
    "Benfica": 0.78, "Sporting CP": 0.76, "Porto": 0.82,
    "Celtic": 0.74, "Rangers": 0.72,
    "Galatasaray": 0.72, "Fenerbahçe": 0.70, "Beşiktaş": 0.65,
    "İstanbul Başakşehir": 0.60,
    "Club Brugge": 0.70, "Anderlecht": 0.65,
    "RB Salzburg": 0.72, "Rapid Wien": 0.60,
    "Slavia Praha": 0.68, "Viktoria Plzeň": 0.62,
    # ── Tier 3 (0.45) — Competitive: top-flight other leagues ─────────────
    "Sunderland": 0.50, "Leeds United": 0.52, "Burnley": 0.50,
    "Sheffield United": 0.48, "Leicester City": 0.60, "Southampton": 0.52,
    "Ipswich Town": 0.48, "Norwich City": 0.48, "Cardiff City": 0.46,
    "Swansea City": 0.46, "Coventry City": 0.46, "Millwall": 0.44,
    "VfL Bochum": 0.55, "Holstein Kiel": 0.52, "Hamburger SV": 0.52,
    "Espanyol": 0.55, "Castellón": 0.45, "Real Oviedo": 0.45,
    "FC Union Berlin": 0.60, "St. Pauli": 0.55,
    "Cremonese": 0.48, "Sassuolo": 0.52, "Bologna": 0.60, "Torino": 0.58,
    "Udinese": 0.52, "Empoli": 0.50, "Hellas Verona": 0.50,
    # ── Tier 4 (0.35) — Other European: Turkish/Dutch/Eredivisie mid ───────
    "Kayserisperi": 0.40, "Trabzonspor": 0.45,
    "Hertha BSC": 0.48,
    # ── Tier 4 (0.40) — Saudi / Middle East ──────────────────────────────
    "Al Hilal": 0.45, "Al Nassr": 0.42, "Al Ahli": 0.40,
    "Al Ittihad": 0.40, "Al Duhail": 0.35, "Al Ahly": 0.38,
    "Al Sadd": 0.33, "Al Wakrah": 0.30, "Al Rayyan": 0.30,
    "Al Qadsiah": 0.30, "Al-Qadsiah": 0.30,
    "Al Hussein": 0.25, "Nasaf": 0.25, "Persepolis": 0.32,
    "Esteghlal": 0.32, "Sepahan": 0.28,
    # ── Tier 5 (0.25) — South American clubs ─────────────────────────────
    "River Plate": 0.42, "Boca Juniors": 0.40, "Flamengo": 0.40,
    "Palmeiras": 0.40, "São Paulo": 0.35, "Atlético Nacional": 0.35,
    "Nacional": 0.35, "Peñarol": 0.33, "Colo-Colo": 0.35,
    "Fluminense": 0.38, "Grêmio": 0.36, "Santos": 0.36,
    "América de Cali": 0.30, "Junior": 0.30, "Millonarios": 0.30,
    "River Plate (Arg)": 0.42, "Vélez Sársfield": 0.32,
    "Independiente": 0.32, "Racing Club": 0.33, "San Lorenzo": 0.30,
    "Estudiantes": 0.30, "Belgrano": 0.28, "Lanús": 0.28,
    "Talleres": 0.28, "Godoy Cruz": 0.28,
    "Club Atlético Tucumán": 0.26, "Olimpia": 0.26, "Libertad": 0.26,
    "Mamelodi Sundowns": 0.35, "Orlando Pirates": 0.32,
    "Kaizer Chiefs": 0.30, "TP Mazembe": 0.30, "Al Ahly (Egy)": 0.35,
    "Zamalek": 0.33, "Wydad AC": 0.30, "Raja Casablanca": 0.30,
    "Auckland": 0.18, "Odense BK": 0.40, "Brøndby": 0.42,
    "Rosenborg": 0.38, "Bodø / Glimt": 0.45,
    "Brann": 0.38, "Vålerenga": 0.35, "Viking": 0.35,
    "IFK Göteborg": 0.38, "Malmö FF": 0.45, "Hammarby": 0.40,
    # ── MLS ───────────────────────────────────────────────────────────────
    "Inter Miami": 0.22, "LA Galaxy": 0.25, "LAFC": 0.25,
    "Seattle Sounders": 0.25, "Portland Timbers": 0.24,
    "Atlanta United": 0.24, "Columbus Crew": 0.24,
    "New England Revolution": 0.22, "NYCFC": 0.23,
    "FC Cincinnati": 0.22, "Minnesota United": 0.22,
    "CF Montréal": 0.22, "Toronto FC": 0.22, "Vancouver Whitecaps": 0.22,
    "Philadelphia Union": 0.23, "D.C. United": 0.22,
    "Orlando City SC": 0.22, "San Jose Earthquakes": 0.20,
    "Houston Dynamo": 0.22,
}

# Default prestige for clubs not in the dict (Big 5 other = 0.55, other Europe = 0.40, rest = 0.28)
CLUB_PRESTIGE_DEFAULT = 0.35

# Minimum effective prestige for chemistry scoring.
# Prevents elite player pairs at low-prestige clubs (e.g. Messi+De Paul at Inter Miami=0.22)
# from scoring near-zero chemistry despite training together daily.
CHEMISTRY_PRESTIGE_FLOOR = 0.55

# Host-country leagues: maps league key → WC2026 host country code
HOST_COUNTRY_LEAGUES: dict[str, str] = {
    "mls":               "usa",
    "canadian_premier":  "can",
    "liga_mx":           "mex",
}

# ---------------------------------------------------------------------------
# Ballon d'Or 2025 — individual excellence scores (normalized 0–1)
# Rank 1 confirmed: Ousmane Dembélé. Rank 2 confirmed: Lamine Yamal.
# Other positions are estimates based on 2024–25 season form up to knowledge cutoff.
# Keys match the exact player names in player_features.parquet.
# Used to compute squad_ballon_dor_score: individual brilliance that club
# prestige and chemistry features don't capture.
# ---------------------------------------------------------------------------
BALLON_DOR_2025: dict[str, float] = {
    "Ousmane Dembélé":     1.000,   # 1st  — France (PSG) [user-confirmed winner]
    "Lamine Yamal":        0.965,   # 2nd  — Spain (FC Barcelona) [user-confirmed]
    "Vinicius Junior":     0.930,   # ~3rd — Brazil (Real Madrid)
    "Jude Bellingham":     0.895,   # ~4th — England (Real Madrid)
    "Kylian Mbappé":       0.860,   # ~5th — France (Real Madrid)
    "Erling Haaland":      0.825,   # ~6th — Norway (Man City)
    "Florian Wirtz":       0.790,   # ~7th — Germany (Bayer Leverkusen)
    "Jamal Musiala":       0.755,   # ~8th — Germany (FC Bayern)
    "Martin Ødegaard":     0.720,   # ~9th — Norway (Arsenal)
    "Bernardo Silva":      0.650,   # ~11th — Portugal (Man City)
    "Phil Foden":          0.685,   # ~10th — England (Man City)
    "Bukayo Saka":         0.615,   # ~12th — England (Arsenal)
    "Harry Kane":          0.580,   # ~13th — England (FC Bayern)
    "Rodri":               0.545,   # ~14th — Spain (Man City; missed 2024–25 with ACL)
    "Pedri":               0.510,   # ~15th — Spain (FC Barcelona)
    "Declan Rice":         0.475,   # ~16th — England (Arsenal)
    "Rúben Dias":          0.440,   # ~17th — Portugal (Man City)
    "Cole Palmer":         0.405,   # ~18th — England (Chelsea)
    "Rafael Leão":         0.370,   # ~19th — Portugal (AC Milan)
    "Bruno Fernandes":     0.335,   # ~20th — Portugal (Man United)
    "Gavi":                0.300,   # ~21st — Spain (FC Barcelona)
    "Dani Olmo":           0.265,   # ~22nd — Spain (FC Barcelona)
    "William Saliba":      0.230,   # ~23rd — France (Arsenal)
    "Fabián Ruiz":         0.195,   # ~24th — Spain (PSG)
    "Alexis Mac Allister": 0.160,   # ~25th — Argentina (Liverpool)
    "Aurélien Tchouaméni": 0.125,   # ~26th — France (Real Madrid)
    "Granit Xhaka":        0.090,   # ~27th — Switzerland (Bayer Leverkusen)
    "Mike Maignan":        0.075,   # ~28th — France (AC Milan)
    "Lionel Messi":        0.055,   # ~29th — Argentina (Inter Miami)
    "Lautaro Martínez":    0.020,   # ~30th — Argentina (Inter)
}

# ---------------------------------------------------------------------------
# Club → league key mapping (used to apply difficulty to RT fallback stats)
# Covers all clubs appearing in the WC2026 squad_master.parquet
# ---------------------------------------------------------------------------
CLUB_LEAGUE_MAP: dict[str, str] = {
    # Premier League
    "Arsenal": "premier_league", "AFC Bournemouth": "premier_league",
    "Aston Villa": "premier_league", "Brentford": "premier_league",
    "Brighton & Hove Albion": "premier_league", "Chelsea": "premier_league",
    "Crystal Palace": "premier_league", "Everton": "premier_league",
    "Fulham": "premier_league", "Ipswich Town": "premier_league",
    "Leicester City": "premier_league", "Liverpool": "premier_league",
    "Manchester City": "premier_league", "Manchester United": "premier_league",
    "Newcastle United": "premier_league", "Nottingham Forest": "premier_league",
    "Southampton": "premier_league", "Tottenham Hotspur": "premier_league",
    "West Ham United": "premier_league", "Wolverhampton Wanderers": "premier_league",
    # English Championship
    "Birmingham City": "english_championship", "Burnley": "english_championship",
    "Cardiff City": "english_championship", "Coventry City": "english_championship",
    "Derby County": "english_championship", "Hull City": "english_championship",
    "Leeds United": "english_championship", "Luton Town": "english_championship",
    "Middlesbrough": "english_championship", "Millwall": "english_championship",
    "Norwich City": "english_championship", "Plymouth Argyle": "english_championship",
    "Portsmouth": "english_championship", "Preston North End": "english_championship",
    "Rotherham United": "english_championship", "Sheffield United": "english_championship",
    "Sheffield Wednesday": "english_championship", "Stoke City": "english_championship",
    "Sunderland": "english_championship", "Swansea City": "english_championship",
    "Watford": "english_championship",
    # English League One
    "Barnsley": "english_league_one", "Charlton Athletic": "english_league_one",
    "Wrexham": "english_league_one", "Port Vale": "english_league_one",
    "Braintree Town": "english_league_one",
    # La Liga
    "Athletic Club": "la_liga", "Atlético Madrid": "la_liga",
    "Celta de Vigo": "la_liga", "FC Barcelona": "la_liga",
    "Girona": "la_liga", "Mallorca": "la_liga",
    "Osasuna": "la_liga", "Rayo Vallecano": "la_liga",
    "Real Betis": "la_liga", "Real Madrid": "la_liga",
    "Real Sociedad": "la_liga", "Sevilla": "la_liga",
    "Valencia": "la_liga", "Villarreal": "la_liga",
    # La Liga 2
    "Castellón": "la_liga", "Elche": "la_liga",
    "Levante": "la_liga", "Racing Santander": "la_liga",
    "Real Oviedo": "la_liga", "Real Zaragoza": "la_liga",
    "Espanyol": "la_liga",
    # Bundesliga
    "Bayer 04 Leverkusen": "bundesliga", "Borussia Dortmund": "bundesliga",
    "Borussia Mönchengladbach": "bundesliga", "Eintracht Frankfurt": "bundesliga",
    "FC Augsburg": "bundesliga", "FC Bayern München": "bundesliga",
    "FC Köln": "bundesliga", "FC Union Berlin": "bundesliga",
    "FSV Mainz 05": "bundesliga", "Holstein Kiel": "bundesliga",
    "RB Leipzig": "bundesliga", "SC Freiburg": "bundesliga",
    "St. Pauli": "bundesliga", "TSG Hoffenheim": "bundesliga",
    "VfB Stuttgart": "bundesliga", "VfL Wolfsburg": "bundesliga",
    "Werder Bremen": "bundesliga",
    # 2. Bundesliga
    "Darmstadt 98": "bundesliga", "Fortuna Düsseldorf": "bundesliga",
    "Hamburger SV": "bundesliga", "Hannover 96": "bundesliga",
    "Karlsruher SC": "bundesliga", "Schalke 04": "bundesliga",
    # Ligue 1
    "Angers SCO": "ligue_1", "Auxerre": "ligue_1",
    "Bastia": "ligue_1", "Guingamp": "ligue_1",
    "Le Havre": "ligue_1", "Lens": "ligue_1",
    "LOSC Lille": "ligue_1", "Lorient": "ligue_1",
    "Monaco": "ligue_1", "Montpellier": "ligue_1",
    "Nancy": "ligue_1", "Nantes": "ligue_1",
    "Nice": "ligue_1", "Olympique Lyonnais": "ligue_1",
    "Olympique Marseille": "ligue_1", "Paris Saint Germain": "ligue_1",
    "Paris": "ligue_1", "Rennes": "ligue_1",
    "Saint-Étienne": "ligue_1", "Sochaux": "ligue_1",
    "Strasbourg": "ligue_1", "Toulouse": "ligue_1",
    "Metz": "ligue_1", "Caen": "ligue_1",
    # Serie A
    "AC Milan": "serie_a", "Atalanta": "serie_a",
    "Bologna": "serie_a", "Cagliari": "serie_a",
    "Como": "serie_a", "Fiorentina": "serie_a",
    "Frosinone": "serie_a", "Genoa": "serie_a",
    "Hellas Verona": "serie_a", "Inter": "serie_a",
    "Juventus": "serie_a", "Lazio": "serie_a",
    "Napoli": "serie_a", "Parma": "serie_a",
    "Roma": "serie_a", "Sampdoria": "serie_a",
    "Sassuolo": "serie_a", "Torino": "serie_a",
    "Udinese": "serie_a", "Venezia": "serie_a",
    "Bari 1908": "serie_a", "Cremonese": "serie_a",
    "Pisa": "serie_a",
    # Eredivisie
    "Ajax": "eredivisie", "Almere City": "eredivisie",
    "AZ": "eredivisie", "FC Den Bosch": "eredivisie",
    "FC Twente": "eredivisie", "FC Utrecht": "eredivisie",
    "FC Volendam": "eredivisie", "Feyenoord": "eredivisie",
    "Heracles Almelo": "eredivisie", "NAC Breda": "eredivisie",
    "NEC Nijmegen": "eredivisie", "PEC Zwolle": "eredivisie",
    "PSV": "eredivisie", "RKC Waalwijk": "eredivisie",
    "SC Cambuur": "eredivisie", "Sparta Rotterdam": "eredivisie",
    "VVV-Venlo": "eredivisie", "Telstar": "eredivisie",
    # Primeira Liga
    "Benfica": "primeira_liga", "Casa Pia": "primeira_liga",
    "Chaves": "primeira_liga", "Estrela Amadora": "primeira_liga",
    "Famalicão": "primeira_liga", "Farense": "primeira_liga",
    "Gil Vicente": "primeira_liga", "Nacional": "primeira_liga",
    "Porto": "primeira_liga", "Sporting Braga": "primeira_liga",
    "Sporting CP": "primeira_liga", "Tondela": "primeira_liga",
    "Torreense": "primeira_liga", "Vitória Guimarães": "primeira_liga",
    "Vizela": "primeira_liga",
    # Turkish Süper Lig
    "Alanyaspor": "turkish_super_lig", "Beşiktaş": "turkish_super_lig",
    "Fenerbahçe": "turkish_super_lig", "Galatasaray": "turkish_super_lig",
    "Gaziantep F.K.": "turkish_super_lig", "Gençlerbirliği": "turkish_super_lig",
    "Göztepe": "turkish_super_lig", "İstanbul Başakşehir": "turkish_super_lig",
    "Kasımpaşa": "turkish_super_lig", "Kayserispor": "turkish_super_lig",
    "Kocaelispor": "turkish_super_lig", "Konyaspor": "turkish_super_lig",
    "Rizespor": "turkish_super_lig", "Sakaryaspor": "turkish_super_lig",
    "Samsunspor": "turkish_super_lig", "Sivasspor": "turkish_super_lig",
    "Trabzonspor": "turkish_super_lig", "Iğdır FK": "turkish_super_lig",
    # Scottish Premiership
    "Celtic": "scottish_premiership", "Hearts": "scottish_premiership",
    "Hibernian": "scottish_premiership", "Kilmarnock": "scottish_premiership",
    "Motherwell": "scottish_premiership", "Rangers": "scottish_premiership",
    # Belgian Pro League
    "Anderlecht": "belgian_pro_league", "Antwerp": "belgian_pro_league",
    "Club Brugge": "belgian_pro_league", "Genk": "belgian_pro_league",
    "Gent": "belgian_pro_league", "Mechelen": "belgian_pro_league",
    "Standard Liège": "belgian_pro_league", "Union Saint-Gilloise": "belgian_pro_league",
    "Zulte-Waregem": "belgian_pro_league", "SK Beveren": "belgian_pro_league",
    "Dender": "belgian_pro_league", "Cercle Brugge": "belgian_pro_league",
    "Sporting Charleroi": "belgian_pro_league",
    # Swiss Super League
    "Lugano": "swiss_super_league", "Servette": "swiss_super_league",
    "St. Gallen": "swiss_super_league", "Young Boys": "swiss_super_league",
    "Zürich": "swiss_super_league",
    # Austrian Bundesliga
    "Grazer AK": "austrian_bundesliga", "LASK Linz": "austrian_bundesliga",
    "Salzburg": "austrian_bundesliga", "Wolfsberger AC": "austrian_bundesliga",
    # Danish Superliga
    "AGF": "danish_superliga", "Brøndby IF": "danish_superliga",
    "FC København": "danish_superliga", "FC Midtjylland": "danish_superliga",
    "Nordsjælland": "danish_superliga", "Odense BK": "danish_superliga",
    "Randers FC": "danish_superliga", "Silkeborg IF": "danish_superliga",
    "Vejle Boldklub": "danish_superliga",
    # Norwegian Eliteserien
    "Bodø / Glimt": "norwegian_eliteserien", "Brann": "norwegian_eliteserien",
    "Molde": "norwegian_eliteserien", "Sarpsborg 08": "norwegian_eliteserien",
    "Viking": "norwegian_eliteserien",
    # Swedish Allsvenskan
    "Malmö FF": "swedish_allsvenskan", "Mjällby": "swedish_allsvenskan",
    # Greek Super League
    "PAOK": "greek_super_league", "Olympiacos F.C.": "greek_super_league",
    "Panathinaikos": "greek_super_league", "Kifisia": "greek_super_league",
    # Cypriot First Division
    "AEK Larnaca": "cypriot_first", "Aris Limassol": "cypriot_first",
    "Omonia Nicosia": "cypriot_first", "Pafos FC": "cypriot_first",
    # Polish Ekstraklasa
    "Cracovia Kraków": "polish_ekstraklasa", "Lechia Gdańsk": "polish_ekstraklasa",
    "Legia Warszawa": "polish_ekstraklasa", "Pogoń Szczecin": "polish_ekstraklasa",
    # Czech First League
    "Hradec Králové": "czech_first_league", "Slavia Praha": "czech_first_league",
    "Slovan Liberec": "czech_first_league", "Sparta Praha": "czech_first_league",
    "Viktoria Plzeň": "czech_first_league",
    # Russian Premier League
    "Dinamo Moskva": "russian_premier", "FK Nizjni Novgorod": "russian_premier",
    "Krasnodar": "russian_premier", "Rostov": "russian_premier",
    "Spartak Moskva": "russian_premier", "Akron": "russian_premier",
    # Serbian SuperLiga
    "Crvena Zvezda": "serbian_super",
    # Croatian Super Liga
    "Dinamo Zagreb": "croatian_super", "Hajduk Split": "croatian_super",
    "Rijeka": "croatian_super", "Slaven Koprivnica": "croatian_super",
    # Hungarian OTP Bank Liga
    "Ferencvárosi": "hungarian_otp", "Győri ETO": "hungarian_otp",
    "Puskás": "hungarian_otp",
    # Romanian Liga I
    "Otelul": "romanian_liga", "Universitatea Cluj": "romanian_liga",
    # Slovenian PrvaLiga
    "Koper": "slovenian_pnl", "Maribor": "slovenian_pnl",
    # Bosnian Premier League
    "Borac Banja Luka": "bosnian_premier",
    # Belarusian Premier League
    "Dinamo Brest": "belarusian_premier",
    # Georgian Erovnuli Liga
    "Dinamo Tbilisi": "georgian_erovnuli",
    # Kazakh Premier League
    "Astana": "kazakh_premier",
    # Uzbek Super League
    "Nasaf": "uzbek_super", "Pakhtakor": "uzbek_super",
    # Slovak Super Liga
    "Slovan Bratislava": "czech_first_league",
    # Bulgarian First League
    "Ludogorets": "bulgarian_first",
    # Israeli Premier League
    "Ashdod": "israeli_premier", "Ironi Kiryat Shmona": "israeli_premier",
    "Maccabi Haifa": "israeli_premier",
    # Irish Premier Division
    "Shamrock Rovers": "irish_premier", "St Patrick's": "irish_premier",
    # Estonian Meistriliiga
    "Paide": "estonian_meistriliiga",
    # Saudi Pro League
    "Abha": "saudi_pro", "Al Ahli": "saudi_pro",
    "Al Ettifaq": "saudi_pro", "Al Fateh": "saudi_pro",
    "Al Hilal": "saudi_pro", "Al Ittihad": "saudi_pro",
    "Al Nassr": "saudi_pro", "Al-Qadsiah": "saudi_pro",
    "Al Riyadh": "saudi_pro", "Al Shabab": "saudi_pro",
    "Al Taawoun": "saudi_pro", "NEOM SC": "saudi_pro",
    "Al-Fayha": "saudi_pro", "Al Wahda": "saudi_pro",
    "Al Nasr": "saudi_pro",
    # UAE Pro League
    "Al Ain": "uae_pro", "Al Jazira": "uae_pro",
    "Bani Yas": "uae_pro", "Dibba Al Fujairah": "uae_pro",
    "Kalba": "uae_pro", "Shabab Al Ahli Dubai": "uae_pro",
    "Sabah": "uae_pro",
    # Qatar Stars League
    "Al Duhail": "qatar_stars", "Al Gharafa": "qatar_stars",
    "Al Hussein": "qatar_stars", "Al Rayyan": "qatar_stars",
    "Al Sadd": "qatar_stars", "Al Sailiya": "qatar_stars",
    "Al Shahaniya": "qatar_stars", "Al Shamal SC": "qatar_stars",
    "Al Wakrah": "qatar_stars", "Al-Ahli Doha": "qatar_stars",
    "Al-Arabi SC": "qatar_stars", "Qatar SC": "qatar_stars",
    "Al Bataeh": "qatar_stars", "Al Najma": "qatar_stars",
    # Iranian Persian Gulf Pro League
    "Aluminium Arak": "iranian_persian_gulf", "Chadormalu SC": "iranian_persian_gulf",
    "Esteghlal": "iranian_persian_gulf", "Foolad": "iranian_persian_gulf",
    "Malavan": "iranian_persian_gulf", "Paykan": "iranian_persian_gulf",
    "Persepolis": "iranian_persian_gulf", "Sepahan": "iranian_persian_gulf",
    "Tractor Sazi": "iranian_persian_gulf",
    # Iraqi Premier League
    "Al Shorta": "iraqi_premier", "Al Zawra'a": "iraqi_premier",
    # Jordanian Premier League
    "Al Wihdat": "jordanian_premier",
    # J1 League
    "Albirex Niigata": "j1_league", "FC Tokyo": "j1_league",
    "Kashima Antlers": "j1_league", "Machida Zelvia": "j1_league",
    # K League
    "Daejeon Citizen": "k_league", "Seoul": "k_league",
    "Gangwon": "k_league", "Jeonbuk Motors": "k_league",
    "Pohang Steelers": "k_league", "Ulsan HD": "k_league",
    # Indonesian Liga 1
    "Persib": "indonesian_liga",
    # Malaysian Super League
    "Selangor": "malaysian_super",
    # A-League (AUS/NZ)
    "Auckland": "a_league", "Melbourne City": "a_league",
    "Melbourne Victory": "a_league", "Newcastle Jets": "a_league",
    "Sydney": "a_league", "Wellington Phoenix": "a_league",
    "Western Sydney Wanderers": "a_league",
    # Brasileirao
    "Athletico PR": "brasileirao", "Atlético Mineiro": "brasileirao",
    "Botafogo": "brasileirao", "Bragantino": "brasileirao",
    "Corinthians": "brasileirao", "Flamengo": "brasileirao",
    "Fluminense": "brasileirao", "Grêmio": "brasileirao",
    "Internacional": "brasileirao", "Palmeiras": "brasileirao",
    "Santos": "brasileirao", "São Paulo": "brasileirao",
    "Vasco da Gama": "brasileirao",
    # Argentine Primera División
    "Boca Juniors": "arg_primera", "Estudiantes": "arg_primera",
    "Huracán": "arg_primera", "Independiente": "arg_primera",
    "Independiente Rivadavia": "arg_primera", "Lanús": "arg_primera",
    "Racing Club": "arg_primera", "River Plate": "arg_primera",
    "Rosario Central": "arg_primera", "San Lorenzo": "arg_primera",
    "Vélez Sarsfield": "arg_primera",
    # Liga MX
    "América": "liga_mx", "Atlas": "liga_mx",
    "Cruz Azul": "liga_mx", "Guadalajara": "liga_mx",
    "Juárez": "liga_mx", "León": "liga_mx",
    "Mazatlán": "liga_mx", "Monterrey": "liga_mx",
    "Pachuca": "liga_mx", "Pumas UNAM": "liga_mx",
    "Tigres UANL": "liga_mx", "Tijuana": "liga_mx",
    "Toluca": "liga_mx",
    # Colombian Primera A
    "Atlético Nacional": "colombian_primera",
    # Chilean Primera División
    "Cobresal": "chilean_primera", "Deportivo Recoleta": "chilean_primera",
    "Universidad Católica": "chilean_primera",
    # Ecuadorian Serie A
    "Barcelona Guayaquil": "ecuadorian_serie_a",
    # Venezuelan Primera División
    "Academia Puerto Cabello": "venezuelan_primera",
    # MLS
    "Atlanta United": "mls", "Austin": "mls",
    "Charlotte": "mls", "Chicago Fire": "mls",
    "Cincinnati": "mls", "Colorado Rapids": "mls",
    "Columbus Crew": "mls", "DC United": "mls",
    "Dallas": "mls", "Inter Miami": "mls",
    "Los Angeles FC": "mls", "Minnesota United": "mls",
    "Nashville SC": "mls", "New England": "mls",
    "New York City": "mls", "New York RB": "mls",
    "Orlando City": "mls", "Philadelphia Union": "mls",
    "Portland Timbers": "mls", "Real Salt Lake": "mls",
    "San Diego": "mls", "Seattle Sounders": "mls",
    "Toronto": "mls", "Vancouver Whitecaps": "mls",
    # Egyptian Premier League
    "Al Ahly": "egyptian_premier", "Ceramica Cleopatra": "egyptian_premier",
    "El Gounah": "egyptian_premier", "Pyramids FC": "egyptian_premier",
    "Zamalek": "egyptian_premier", "ZED FC": "egyptian_premier",
    "Al Masry": "egyptian_premier",
    # Moroccan Botola Pro
    "FAR Rabat": "moroccan_botola", "Raja Casablanca": "moroccan_botola",
    "RSB Berkane": "moroccan_botola", "Wydad Casablanca": "moroccan_botola",
    # Algerian Ligue Professionnelle 1
    "JS Kabylie": "algerian_ligue",
    # Tunisian Ligue 1
    "ES Tunis": "tunisian_ligue", "US Monastir": "tunisian_ligue",
    # South African Premier Soccer League
    "Kaizer Chiefs": "south_african_psl", "Mamelodi Sundowns": "south_african_psl",
    "Orlando Pirates": "south_african_psl", "Polokwane City": "south_african_psl",
    "Sekhukhune United": "south_african_psl", "Siwelele": "south_african_psl",
}


def _weighted_mean(series: pd.Series, weights: list) -> float:
    """Compute weighted mean of last N values (most recent first)."""
    vals = series.dropna().tail(len(weights)).values[::-1]
    w = weights[:len(vals)]
    if sum(w) == 0 or len(vals) == 0:
        return np.nan
    return float(np.dot(vals, w) / sum(w))


# ---------------------------------------------------------------------------
# Stream 1: Club features
# ---------------------------------------------------------------------------

def build_club_features() -> pd.DataFrame:
    """
    Aggregate club season data into per-player rolling features.
    Returns one row per player with their club-level stats.
    """
    club_path = RAW_CLUB / "all_club_seasons.parquet"
    if not club_path.exists():
        print("[WARN] Club seasons data not found. Run ingest/club_seasons.py first.")
        return pd.DataFrame()

    df = pd.read_parquet(club_path)

    # Numeric coerce
    for col in ["xg", "xag", "npxg", "min", "goals", "assists",
                "progressive_carries", "progressive_passes", "touches_att_pen_area",
                "pressures", "pressure_regains"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter out rows with too few minutes
    if "min" in df.columns:
        df = df[df["min"] >= CFG["min_club_minutes_per_season"]]

    # Per-90 calculation
    df["p90"] = df["min"] / 90
    for stat in ["xg", "xag", "npxg", "goals", "assists", "shots", "shots_on_target",
                 "progressive_carries", "progressive_passes",
                 "touches_att_pen_area", "pressure_regains"]:
        if stat in df.columns:
            df[f"{stat}_p90"] = df[stat] / df["p90"].replace(0, np.nan)

    # Use xG if available, fall back to goals as primary attacking signal
    attack_col = "xg_p90" if "xg_p90" in df.columns else "goals_p90"
    assist_col  = "xag_p90" if "xag_p90" in df.columns else "assists_p90"

    # League difficulty adjustment
    if "league_name" in df.columns and attack_col in df.columns:
        df["league_difficulty"] = df["league_name"].map(LEAGUE_DIFFICULTY).fillna(0.75)
        df["attack_p90_adj"] = df[attack_col] * df["league_difficulty"]
    elif attack_col in df.columns:
        df["attack_p90_adj"] = df[attack_col]

    df = df.sort_values(["player", "season"])

    # Build agg spec from whatever columns actually exist
    agg_map = {
        "club_xg_p90":                  ("attack_p90_adj",         "mean"),
        "club_xa_p90":                  (assist_col,               "mean"),
        "club_goals_p90":               ("goals_p90",              "mean"),
        "club_shots_p90":               ("shots_p90",              "mean"),
        "club_shots_on_target_p90":     ("shots_on_target_p90",    "mean"),
        "club_progressive_carries_p90": ("progressive_carries_p90","mean"),
        "club_pressure_regains_p90":    ("pressure_regains_p90",   "mean"),
        "seasons_in_data":              ("season",                 "count"),
        "primary_league":               ("league_name", lambda x: x.mode()[0] if not x.empty else None),
        "primary_club":                 ("team",                   "last"),
    }
    agg_spec = {k: v for k, v in agg_map.items() if v[0] in df.columns}
    agg = df.groupby("player").agg(**agg_spec).reset_index()

    # Weighted rolling form (most recent season = highest weight)
    form_src = "attack_p90_adj" if "attack_p90_adj" in df.columns else "goals_p90"
    def rolling_attack(player_df):
        return _weighted_mean(player_df.sort_values("season")[form_src], FORM_WEIGHTS)

    weighted_form = (
        df.groupby("player")
        .apply(rolling_attack)
        .rename("club_xg_rolling_weighted")
        .reset_index()
    )
    agg = agg.merge(weighted_form, on="player", how="left")

    return agg


# ---------------------------------------------------------------------------
# Stream 2: International features
# ---------------------------------------------------------------------------

def build_intl_features() -> pd.DataFrame:
    """
    Aggregate international player events into per-player features.
    Tournament weights from config are applied before averaging.
    """
    events_path = RAW_INTL / "all_international_player_events.parquet"
    if not events_path.exists():
        print("[WARN] International events not found. Run ingest/international.py first.")
        return pd.DataFrame()

    df = pd.read_parquet(events_path)

    for col in ["xg", "goals", "shots", "key_passes", "assists", "tournament_weight"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Weight each row by tournament importance
    for col in ["xg", "goals", "shots", "key_passes"]:
        if col in df.columns and "tournament_weight" in df.columns:
            df[f"{col}_weighted"] = df[col] * df["tournament_weight"]

    agg = df.groupby("player").agg(
        intl_matches=("match_id", "nunique"),
        intl_xg_total=("xg_weighted", "sum"),
        intl_goals_total=("goals_weighted", "sum"),
        intl_shots_total=("shots_weighted", "sum"),
        intl_key_passes_total=("key_passes_weighted", "sum"),
        intl_teams=("team", lambda x: x.mode()[0] if not x.empty else None),
    ).reset_index()

    # Per-match averages (proxy for per-90 at international level)
    for stat in ["xg", "goals", "shots", "key_passes"]:
        col = f"intl_{stat}_total"
        if col in agg.columns:
            agg[f"intl_{stat}_per_match"] = agg[col] / agg["intl_matches"].clip(lower=1)

    # Filter players with too few caps for form to be meaningful
    agg["intl_form_reliable"] = agg["intl_matches"] >= MIN_CAPS

    return agg


# ---------------------------------------------------------------------------
# Stream 3 + 4: Squad + Coach features
# ---------------------------------------------------------------------------

def build_squad_coach_features() -> pd.DataFrame:
    """Load WC2026 squad master + coach history and merge."""
    squad_path = RAW_SQUADS / "squad_master.parquet"
    coach_path = RAW_COACH / "coach_player_history.parquet"

    if not squad_path.exists():
        print("[WARN] squad_master.parquet not found. Run ingest/squads.py first.")
        return pd.DataFrame()

    squads = pd.read_parquet(squad_path)

    if coach_path.exists():
        coach = pd.read_parquet(coach_path)
        merge_cols = [c for c in ["player", "team"] if c in squads.columns and c in coach.columns]
        squads = squads.merge(coach, on=merge_cols, how="left", suffixes=("", "_coach"))
    else:
        print("[WARN] Coach history not found. Run ingest/coach_history.py first.")
        squads["coach_familiarity_score"] = np.nan
        squads["coach_familiarity_tier"] = "unknown"

    return squads


# ---------------------------------------------------------------------------
# RT fallback: fill club features from squad_master risingtransfers stats
# Used for all players NOT covered by FBref Big 5 pull
# ---------------------------------------------------------------------------

def fill_rt_club_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    For players where FBref club data is missing (club_xg_p90 is NaN),
    fill using squad_master RT per-90 stats scaled by league difficulty.

    squad_master already has goals_per90, assists_per90, shots_per90 etc.
    from risingtransfers, covering all leagues globally.
    """
    df = features.copy()

    # Identify players with no FBref club data
    needs_fill = df["club_xg_p90"].isna() if "club_xg_p90" in df.columns else pd.Series(True, index=df.index)

    if needs_fill.sum() == 0:
        return df

    # Map each player's club to league difficulty
    if "club" not in df.columns:
        return df

    def get_difficulty(club):
        league = CLUB_LEAGUE_MAP.get(club)
        if league is None:
            return 0.68  # conservative default for unknown leagues
        return LEAGUE_DIFFICULTY.get(league, 0.68)

    df["_rt_difficulty"] = df["club"].map(get_difficulty)

    # Fill club_xg_p90: use goals_per90 * league_difficulty
    if "goals_per90" in df.columns:
        rt_xg = df["goals_per90"] * df["_rt_difficulty"]
        if "club_xg_p90" not in df.columns:
            df["club_xg_p90"] = np.nan
        df.loc[needs_fill, "club_xg_p90"] = rt_xg[needs_fill]

    # Fill club_xa_p90: use assists_per90 * league_difficulty
    if "assists_per90" in df.columns:
        rt_xa = df["assists_per90"] * df["_rt_difficulty"]
        if "club_xa_p90" not in df.columns:
            df["club_xa_p90"] = np.nan
        xa_needs_fill = df["club_xa_p90"].isna()
        df.loc[xa_needs_fill, "club_xa_p90"] = rt_xa[xa_needs_fill]

    # Fill club_goals_p90 from RT directly (no difficulty scaling — raw stat)
    if "goals_per90" in df.columns:
        if "club_goals_p90" not in df.columns:
            df["club_goals_p90"] = np.nan
        g_needs_fill = df["club_goals_p90"].isna()
        df.loc[g_needs_fill, "club_goals_p90"] = df["goals_per90"][g_needs_fill]

    # Fill club_shots_p90 from RT shots_per90
    if "shots_per90" in df.columns:
        if "club_shots_p90" not in df.columns:
            df["club_shots_p90"] = np.nan
        s_needs_fill = df["club_shots_p90"].isna()
        df.loc[s_needs_fill, "club_shots_p90"] = df["shots_per90"][s_needs_fill]

    # club_xg_rolling_weighted: use RT goals_per90 * difficulty as best proxy
    if "goals_per90" in df.columns:
        rt_rolling = df["goals_per90"] * df["_rt_difficulty"]
        if "club_xg_rolling_weighted" not in df.columns:
            df["club_xg_rolling_weighted"] = np.nan
        rw_needs_fill = df["club_xg_rolling_weighted"].isna()
        df.loc[rw_needs_fill, "club_xg_rolling_weighted"] = rt_rolling[rw_needs_fill]

    filled = needs_fill.sum()
    total = len(df)
    print(f"  RT fallback filled {filled}/{total} players "
          f"({filled/total*100:.0f}%) from squad_master stats")

    df = df.drop(columns=["_rt_difficulty"], errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Delta features — the club-to-international gap
# ---------------------------------------------------------------------------

def compute_delta_features(player_features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the club-to-international performance delta.

    club_to_intl_xg_delta > 0 means player outperforms at club vs international.
    Large positive delta = potential underperformance risk at WC (the Raphinha signal).
    Negative delta = player elevates for international football.
    """
    df = player_features.copy()

    if "club_xg_p90" in df.columns and "intl_xg_per_match" in df.columns:
        df["club_to_intl_xg_delta"] = df["club_xg_p90"] - df["intl_xg_per_match"]

    if "club_xa_p90" in df.columns and "intl_key_passes_per_match" in df.columns:
        df["club_to_intl_creation_delta"] = df["club_xa_p90"] - df["intl_key_passes_per_match"]

    # Discretize delta into risk tiers
    if "club_to_intl_xg_delta" in df.columns:
        df["intl_transfer_risk"] = pd.cut(
            df["club_to_intl_xg_delta"],
            bins=[-np.inf, -0.05, 0.05, 0.15, np.inf],
            labels=["elevates", "neutral", "slight_drop", "significant_drop"],
        )

    return df


# ---------------------------------------------------------------------------
# Team-level aggregation
# ---------------------------------------------------------------------------

def build_team_features(player_features: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player features up to team level for the match model.
    Only aggregates columns that actually exist — safe to run with partial data.
    """
    if player_features.empty or "team" not in player_features.columns:
        return pd.DataFrame()

    cols = player_features.columns.tolist()

    # Build agg spec dynamically — only include columns present in the data
    agg_spec = {"n_players": ("player", "count")}

    if "club_xg_p90" in cols:
        agg_spec["squad_avg_club_xg_p90"] = ("club_xg_p90", "mean")
    if "intl_xg_per_match" in cols:
        agg_spec["squad_avg_intl_xg"] = ("intl_xg_per_match", "mean")
    if "coach_familiarity_score" in cols:
        agg_spec["squad_avg_coach_familiarity"] = ("coach_familiarity_score", "mean")
    if "xi_coach_overlap_pct" in cols:
        agg_spec["xi_coach_overlap_pct"] = ("xi_coach_overlap_pct", "first")
    if "intl_matches" in cols:
        agg_spec["avg_intl_matches"] = ("intl_matches", "mean")
    if "coach_familiarity_tier" in cols:
        agg_spec["pct_high_familiarity"] = (
            "coach_familiarity_tier",
            lambda x: (x == "high").sum() / len(x) if len(x) > 0 else 0
        )
    if "intl_transfer_risk" in cols:
        agg_spec["pct_significant_drop"] = (
            "intl_transfer_risk",
            lambda x: (x == "significant_drop").sum() / len(x) if len(x) > 0 else 0
        )

    agg = player_features.groupby("team").agg(**agg_spec).reset_index()

    # Position-weighted attack quality: FW=4, MF=2, DF=1, GK=0
    # Better captures star-forward impact (e.g. Messi) vs simple squad average
    if "club_xg_p90" in cols and "position" in cols:
        POS_WEIGHT = {"FW": 4, "MF": 2, "DF": 1, "GK": 0}

        def pos_weighted_xg(df):
            w = df["position"].map(POS_WEIGHT).fillna(1).astype(float)
            total_w = w.sum()
            if total_w == 0:
                return np.nan
            xg = df["club_xg_p90"].fillna(0)
            return float((xg * w).sum() / total_w)

        pw_xg = (
            player_features.groupby("team")
            .apply(pos_weighted_xg)
            .reset_index()
            .rename(columns={0: "squad_attack_quality"})
        )
        agg = agg.merge(pw_xg, on="team", how="left")

    # Position-weighted club prestige: FW=3, MF=2, DF=1, GK=0
    # Captures trophy culture + UCL exposure; higher for squads at elite clubs.
    # Messi (Inter Miami=0.22 FW) vs Kane (Bayern=1.0 FW) → Argentina correctly docked here.
    if "club" in cols and "position" in cols:
        POS_WEIGHT_P = {"FW": 3, "MF": 2, "DF": 1, "GK": 0}

        def pos_weighted_prestige(df):
            w = df["position"].map(POS_WEIGHT_P).fillna(1).astype(float)
            total_w = w.sum()
            if total_w == 0:
                return np.nan
            p = df["club"].map(CLUB_PRESTIGE).fillna(CLUB_PRESTIGE_DEFAULT)
            return float((p * w).sum() / total_w)

        pw_pres = (
            player_features.groupby("team")
            .apply(pos_weighted_prestige)
            .reset_index()
            .rename(columns={0: "squad_club_prestige"})
        )
        agg = agg.merge(pw_pres, on="team", how="left")

    # Club chemistry: pairs of national team players who already play together
    # at the same club. A FW+MF pair from Barcelona/PSG have pre-built patterns,
    # runs, and pressing triggers. C(n,2) pairs × prestige × avg position weight.
    # Normalised by squad_size² so raw headcount doesn't dominate.
    if "club" in cols and "position" in cols:
        POS_W_CHEM = {"FW": 3, "MF": 2, "DF": 1, "GK": 0.5}

        def club_chemistry_score(df):
            total = 0.0
            n = len(df)
            if n < 2:
                return 0.0
            for club_name, grp in df.groupby("club"):
                if len(grp) < 2:
                    continue
                raw_prestige = CLUB_PRESTIGE.get(club_name, CLUB_PRESTIGE_DEFAULT)
                # Floor prevents elite pairs (e.g. Messi+De Paul at Inter Miami=0.22)
                # from scoring near-zero chemistry despite training together daily.
                prestige = max(raw_prestige, CHEMISTRY_PRESTIGE_FLOOR)
                pos_weights = grp["position"].map(POS_W_CHEM).fillna(1).tolist()
                # Sum all unique pairs
                for i in range(len(pos_weights)):
                    for j in range(i + 1, len(pos_weights)):
                        pair_w = (pos_weights[i] + pos_weights[j]) / 2.0
                        total += pair_w * prestige
            return total / (n * n)  # normalise

        chem = (
            player_features.groupby("team")
            .apply(club_chemistry_score)
            .reset_index()
            .rename(columns={0: "squad_club_chemistry"})
        )
        agg = agg.merge(chem, on="team", how="left")

    # Individual brilliance: best-player Ballon d'Or 2025 score in the squad.
    # Uses MAX (not sum) to capture "does this team have a genuine superstar?"
    # — Yamal (0.965) vs Bellingham (0.895) vs Bernardo Silva (0.650).
    # 0–1 range keeps the feature well-behaved in the model.
    if "player" in player_features.columns:
        def ballon_dor_squad_score(df):
            # Depth-aware individual-brilliance score: a geometric-decay weighted
            # sum over the WHOLE squad's Ballon d'Or 2025 rankings — full weight on
            # the top-ranked player, half on the 2nd, quarter on the 3rd, ... So a
            # squad stacked with elite talent (e.g. France: Dembélé #1 + Mbappé #5)
            # scores clearly above one with a single star. Tree model is
            # scale-invariant, so the (>1) range is fine.
            scores = sorted(
                (BALLON_DOR_2025.get(str(p), 0.0) for p in df["player"]),
                reverse=True,
            )
            return sum(s * (0.5 ** i) for i, s in enumerate(scores))

        bdo = (
            player_features.groupby("team")
            .apply(ballon_dor_squad_score)
            .reset_index()
            .rename(columns={0: "squad_ballon_dor_score"})
        )
        agg = agg.merge(bdo, on="team", how="left")
        agg["squad_ballon_dor_score"] = agg["squad_ballon_dor_score"].fillna(0.0)

    # Host-country familiarity: position-weighted fraction of squad currently
    # playing in MLS (USA), Liga MX (MEX), or CPL (CAN).
    # Applied as a squad feature so the model/post-prediction boost can use it.
    if "club" in cols and "position" in cols:
        POS_W_HOST = {"FW": 3, "MF": 2, "DF": 1, "GK": 0.5}

        for league_key, country_code in HOST_COUNTRY_LEAGUES.items():
            col_name = f"host_familiarity_{country_code}"

            def host_fam_score(df, _lk=league_key):
                w = df["position"].map(POS_W_HOST).fillna(1).astype(float)
                total_w = w.sum()
                if total_w == 0:
                    return 0.0
                in_host = df["club"].map(
                    lambda c: CLUB_LEAGUE_MAP.get(c, "") == _lk
                ).astype(float)
                return float((in_host * w).sum() / total_w)

            hf = (
                player_features.groupby("team")
                .apply(host_fam_score)
                .reset_index()
                .rename(columns={0: col_name})
            )
            agg = agg.merge(hf, on="team", how="left")
            agg[col_name] = agg[col_name].fillna(0.0)

    missing = [c for c in ["squad_avg_club_xg_p90", "squad_attack_quality",
                            "squad_club_prestige", "squad_club_chemistry"]
               if c not in agg.columns]
    if missing:
        print(f"  [NOTE] Team features built without: {missing} "
              f"(run club seasons pull to add these)")

    return agg


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_all_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build player_features and team_features parquet files.
    Returns (player_features, team_features).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Building club features...")
    club = build_club_features()

    print("Building international features...")
    intl = build_intl_features()

    print("Building squad + coach features...")
    squad_coach = build_squad_coach_features()

    # Merge streams: squad is the spine (only WC2026 players matter)
    if squad_coach.empty:
        print("[ERROR] No squad data — cannot build features.")
        return pd.DataFrame(), pd.DataFrame()

    player_col = "player" if "player" in squad_coach.columns else "name"
    squad_coach = squad_coach.rename(columns={player_col: "player"})

    features = squad_coach.copy()

    if not club.empty:
        features = features.merge(club, on="player", how="left")

    # Fill gaps with RT stats from squad_master for players not in Big 5
    print("Filling club feature gaps from risingtransfers stats...")
    features = fill_rt_club_features(features)

    if not intl.empty:
        features = features.merge(intl, on="player", how="left")

    features = compute_delta_features(features)

    # Normalize team names to WC2026 display names before aggregating
    if "team" in features.columns:
        features["team"] = features["team"].replace(TEAM_NAME_NORMALIZE)

    team_features = build_team_features(features)

    if "team" in team_features.columns:
        team_features["team"] = team_features["team"].replace(TEAM_NAME_NORMALIZE)

    features.to_parquet(PROCESSED_DIR / "player_features.parquet", index=False)
    team_features.to_parquet(PROCESSED_DIR / "team_features.parquet", index=False)

    print(f"\nFeatures built:")
    print(f"  player_features: {len(features)} rows, {len(features.columns)} cols")
    print(f"  team_features:   {len(team_features)} rows, {len(team_features.columns)} cols")

    return features, team_features


if __name__ == "__main__":
    build_all_features()
