"""
XGBoost match outcome model.

Input:  team-pair feature vector (differences between two teams)
Output: P(home_win), P(draw), P(away_win)

Training data: WC 2010-2022 matches + qualifier results from StatsBomb.
Calibration:   isotonic regression post-training.

GPU training: set device='cuda' if running on the Amaral cluster.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, brier_score_loss
import xgboost as xgb
import yaml
import pickle

CFG = yaml.safe_load(open(Path(__file__).parents[1] / "configs/config.yaml"))
PROCESSED_DIR = Path(__file__).parents[1] / CFG["paths"]["processed"]
MODEL_DIR = Path(__file__).parents[1] / "outputs"
MODEL_DIR.mkdir(exist_ok=True)

# Features used in match prediction
# All expressed as (home_team - away_team) differences
MATCH_FEATURES = [
    "elo_diff",
    "net_xg_diff",                   # rolling weighted xG - xGA
    "squad_avg_club_xg_diff",        # squad average club quality
    "squad_attack_quality_diff",     # position-weighted attack quality (FW=4x, MF=2x, DF=1x)
    "squad_club_prestige_diff",      # position-weighted club trophy tier (UCL/silverware culture)
    "squad_club_chemistry_diff",     # same-club pairs in national squad: pre-built patterns transfer
    "avg_intl_matches_diff",         # experience proxy
    "pct_significant_drop_diff",     # club-to-intl transfer risk
    "is_knockout",                   # 1 if knockout stage
    "squad_ballon_dor_diff",         # sum of Ballon d'Or 2025 scores — individual brilliance (Yamal, Haaland…)
]

LABEL_MAP = {"home_win": 0, "draw": 1, "away_win": 2}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}


# ---------------------------------------------------------------------------
# ELO ratings — simple implementation seeded from World Football ELO
# ---------------------------------------------------------------------------

ELO_INITIAL = 1500
ELO_K = 32

# Pre-tournament ELO ratings from worldcupelo.com (June 2026).
# These override the historical ELO for all 48 WC2026 teams in live simulation.
WC2026_SEEDED_ELO: dict[str, float] = {
    "Spain":                   2171,
    "Argentina":               2113,
    "France":                  2063,
    "England":                 2042,
    "Colombia":                1998,
    "Brazil":                  1979,
    "Portugal":                1976,
    "Netherlands":             1959,
    "Croatia":                 1933,
    "Ecuador":                 1933,
    "Germany":                 1910,
    "Norway":                  1922,
    "Switzerland":             1897,
    "Uruguay":                 1890,
    "Turkey":                  1880,
    "Senegal":                 1869,
    "Belgium":                 1849,
    "Mexico":                  1834,
    "Paraguay":                1833,
    "Austria":                 1818,
    "Canada":                  1806,
    "Morocco":                 1806,
    "Scotland":                1790,
    "South Korea":             1784,
    "Australia":               1774,
    "Iran":                    1754,
    "USA":                     1747,
    "Algeria":                 1728,
    "Uzbekistan":              1735,
    "Czechia":                 1731,
    "Panama":                  1743,
    "Japan":                   1879,
    "Sweden":                  1660,
    "Egypt":                   1660,
    "Ivory Coast":             1637,
    "DR Congo":                1639,
    "Jordan":                  1691,
    "Bosnia and Herzegovina":  1571,
    "Cabo Verde":              1561,
    "New Zealand":             1586,
    "Saudi Arabia":            1592,
    "South Africa":            1529,
    "Ghana":                   1509,
    "Iraq":                    1583,
    "Tunisia":                 1614,
    "Haiti":                   1542,
    "Qatar":                   1427,
    "Curaçao":                 1467,
}

def _resolve_name(row, side: str) -> str:
    """Robust team-name lookup.

    On the StatsBomb major-tournament rows the `{side}_team_name` column exists
    but is null, while `{side}_team` is populated. A plain
    `row.get("home_team_name", row.get("home_team"))` returns the null instead of
    falling back (the key exists), which silently dropped 314 WC/Euro/Copa/AFCON
    matches. This falls back correctly.
    """
    v = row.get(f"{side}_team_name")
    if not isinstance(v, str) or not v.strip():
        v = row.get(f"{side}_team", "")
    return v if isinstance(v, str) else ""


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def build_elo_ratings(matches: pd.DataFrame) -> dict[str, float]:
    """
    Build ELO ratings from historical match results.
    matches must have: home_team_name, away_team_name, home_score, away_score, date
    Sorted chronologically.
    """
    ratings = {}

    def get(team):
        return ratings.get(team, ELO_INITIAL)

    date_col = "date" if "date" in matches.columns else "match_date"
    for _, row in matches.sort_values(date_col).iterrows():
        home = _resolve_name(row, "home")
        away = _resolve_name(row, "away")
        if not home or not away:
            continue

        h_score = pd.to_numeric(row.get("home_score", 0), errors="coerce") or 0
        a_score = pd.to_numeric(row.get("away_score", 0), errors="coerce") or 0

        h_rating, a_rating = get(home), get(away)
        h_expected = _elo_expected(h_rating, a_rating)

        if h_score > a_score:
            h_actual, a_actual = 1.0, 0.0
        elif h_score == a_score:
            h_actual, a_actual = 0.5, 0.5
        else:
            h_actual, a_actual = 0.0, 1.0

        w = row.get("tournament_weight", 0.5)
        k = ELO_K * (1 + w)  # higher-stakes matches update ELO more

        ratings[home] = h_rating + k * (h_actual - h_expected)
        ratings[away] = a_rating + k * (a_actual - (1 - h_expected))

    return ratings


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def build_match_features(matches: pd.DataFrame,
                          team_features: pd.DataFrame,
                          elo_ratings: dict) -> pd.DataFrame:
    """
    For each match, compute the feature difference vector (home - away).
    Returns X (features) and y (labels) ready for XGBoost.
    """
    tf = team_features.set_index("team") if "team" in team_features.columns else team_features

    rows = []
    for _, m in matches.iterrows():
        home = _resolve_name(m, "home")
        away = _resolve_name(m, "away")

        if not home or not away:
            continue

        h = tf.loc[home] if home in tf.index else pd.Series(dtype=float)
        a = tf.loc[away] if away in tf.index else pd.Series(dtype=float)

        h_score = pd.to_numeric(m.get("home_score", np.nan), errors="coerce")
        a_score = pd.to_numeric(m.get("away_score", np.nan), errors="coerce")

        if pd.isna(h_score) or pd.isna(a_score):
            label = np.nan  # prediction mode — no label, row still included
        elif h_score > a_score:
            label = "home_win"
        elif h_score == a_score:
            label = "draw"
        else:
            label = "away_win"

        def diff(col):
            hv = pd.to_numeric(h.get(col, np.nan), errors="coerce")
            av = pd.to_numeric(a.get(col, np.nan), errors="coerce")
            if pd.isna(hv) or pd.isna(av):
                return np.nan
            return hv - av

        # Net xG = rolling weighted xG minus xGA
        h_net_xg = (pd.to_numeric(h.get("squad_avg_club_xg_p90", np.nan), errors="coerce") or 0) - \
                   (pd.to_numeric(h.get("squad_avg_intl_xg", np.nan), errors="coerce") or 0)
        a_net_xg = (pd.to_numeric(a.get("squad_avg_club_xg_p90", np.nan), errors="coerce") or 0) - \
                   (pd.to_numeric(a.get("squad_avg_intl_xg", np.nan), errors="coerce") or 0)

        row = {
            "home_team": home,
            "away_team": away,
            "date": m.get("date", ""),
            "tournament_name": m.get("tournament_name", ""),
            "label": label,
            "elo_diff": elo_ratings.get(home, ELO_INITIAL) - elo_ratings.get(away, ELO_INITIAL),
            "net_xg_diff": h_net_xg - a_net_xg,
            "squad_avg_club_xg_diff": diff("squad_avg_club_xg_p90"),
            "squad_attack_quality_diff": diff("squad_attack_quality"),
            "squad_club_prestige_diff": diff("squad_club_prestige"),
            "squad_club_chemistry_diff": diff("squad_club_chemistry"),
            "squad_avg_coach_familiarity_diff": diff("squad_avg_coach_familiarity"),
            "xi_coach_overlap_diff": diff("xi_coach_overlap_pct"),
            "avg_intl_matches_diff": diff("avg_intl_matches"),
            "pct_high_familiarity_diff": diff("pct_high_familiarity"),
            "pct_significant_drop_diff": diff("pct_significant_drop"),
            "rest_days_diff": pd.to_numeric(m.get("rest_days_diff", 0), errors="coerce") or 0,
            "squad_ballon_dor_diff": diff("squad_ballon_dor_score"),
            "is_knockout": int((pd.to_numeric(m.get("match_week", 0), errors="coerce") or 0) > 3 or
                               str(m.get("competition_stage_name", "") or "").lower() in
                               ("round of 16", "quarter-final", "semi-final", "final",
                                "third place")),
            "is_neutral_venue": int(bool(m.get("neutral"))) if m.get("neutral") is not None else 1,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train(use_gpu: bool = False) -> tuple:
    """
    Train XGBoost match outcome classifier.

    use_gpu=True: set device='cuda' for Amaral cluster.
    Returns (model, calibrated_model, label_encoder, feature_cols).
    """
    matches_path = Path(__file__).parents[1] / CFG["paths"]["raw_international"] / \
                   "all_international_matches.parquet"
    team_features_path = PROCESSED_DIR / "team_features.parquet"

    if not matches_path.exists() or not team_features_path.exists():
        print("[ERROR] Run run_pipeline.py first to generate data.")
        return None, None, None, None

    matches = pd.read_parquet(matches_path)
    team_features = pd.read_parquet(team_features_path)

    print("Building ELO ratings...")
    elo = build_elo_ratings(matches)

    print("Building match feature matrix...")
    match_df = build_match_features(matches, team_features, elo)
    match_df = match_df[match_df["label"].notna()].reset_index(drop=True)

    feature_cols = [c for c in MATCH_FEATURES if c in match_df.columns]
    X = match_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y_raw = match_df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"Training on {len(X)} matches, {len(feature_cols)} features")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    xgb_params = {
        "n_estimators":     CFG["model"]["n_estimators"],
        "max_depth":        CFG["model"]["max_depth"],
        "learning_rate":    CFG["model"]["learning_rate"],
        "subsample":        CFG["model"]["subsample"],
        "colsample_bytree": CFG["model"]["colsample_bytree"],
        "random_state":     CFG["model"]["random_state"],
        "reg_lambda":       5.0,   # L2 regularization — prevents extreme splits on sparse features
        "min_child_weight": 4,     # leaf needs ≥4 training examples (avoids 1-sample spurious splits)
        "objective":        "multi:softprob",
        "num_class":        3,
        "eval_metric":      "mlogloss",
        "device":           "cuda" if use_gpu else "cpu",
    }

    clf = xgb.XGBClassifier(**xgb_params)

    # Step 1: get out-of-fold probability predictions for calibration
    # (sklearn 1.7 broke CalibratedClassifierCV with XGBoost — manual approach instead)
    cv = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=CFG["model"]["random_state"])

    print("Fitting XGBoost (5-fold OOF for calibration)...")
    # Manual OOF loop avoids sklearn 1.6/xgboost 2.0 __sklearn_tags__ incompatibility
    oof_probs = np.zeros((len(X), len(np.unique(y))))
    for train_idx, val_idx in cv.split(X, y):
        clf_fold = xgb.XGBClassifier(**xgb_params)
        clf_fold.fit(X[train_idx], y[train_idx])
        oof_probs[val_idx] = clf_fold.predict_proba(X[val_idx])

    # Step 2: fit isotonic calibrators per class on OOF predictions
    calibrators = []
    for c in range(3):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_probs[:, c], (y == c).astype(float))
        calibrators.append(iso)

    # Step 3: fit final model on all data
    clf.fit(X, y)

    # Evaluation: apply calibration to full-data predictions
    raw_probs = clf.predict_proba(X)
    cal_probs = np.column_stack([calibrators[c].predict(raw_probs[:, c]) for c in range(3)])
    cal_probs = cal_probs / cal_probs.sum(axis=1, keepdims=True)  # renormalise

    ll_raw = log_loss(y, raw_probs)
    ll_cal = log_loss(y, cal_probs)
    print(f"  Train log-loss (raw):       {ll_raw:.4f}")
    print(f"  Train log-loss (calibrated):{ll_cal:.4f}  (random = {np.log(3):.4f})")

    # Feature importance
    importance = dict(zip(feature_cols, clf.feature_importances_))
    print("\nFeature importance:")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat:<45} {imp:.4f}")

    # Save: store base clf + calibrators together
    model_path = MODEL_DIR / "match_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "calibrators": calibrators,
                     "label_encoder": le, "feature_cols": feature_cols, "elo": elo}, f)
    print(f"\nModel saved → {model_path}")

    return clf, le, feature_cols, elo


def load_model() -> tuple:
    """Load saved model, label encoder, feature cols, ELO ratings, and calibrators."""
    model_path = MODEL_DIR / "match_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("No trained model found. Run model.train() first.")
    with open(model_path, "rb") as f:
        d = pickle.load(f)
    return d["model"], d["label_encoder"], d["feature_cols"], d["elo"], d.get("calibrators")


def update_elo_with_live_results(elo: dict, wc_matches: list[dict],
                                  name_map: dict | None = None,
                                  tournament_weight: float = 1.0,
                                  seed_elo: dict | None = None) -> dict:
    """
    Apply WC2026 completed match results to an existing ELO dict.

    1. Seed with real-world pre-tournament ELO (from worldcupelo.com) if provided.
    2. Normalise ELO keys using name_map (StatsBomb → WC2026 display names).
    3. Apply each completed match with the given tournament_weight.

    wc_matches: list of dicts with keys: home, away, hg, ag
    name_map: {old_name: new_name} — applied to ELO keys only
    seed_elo: {team_name: elo} — pre-tournament ELO that overrides historical for WC2026 teams
    """
    # Step 1: normalise ELO key names
    updated = {}
    for team, rating in elo.items():
        canonical = name_map.get(team, team) if name_map else team
        # Keep higher rating if both forms exist (shouldn't happen but be safe)
        if canonical in updated:
            updated[canonical] = max(updated[canonical], rating)
        else:
            updated[canonical] = rating

    # Step 2: override with real-world seed ELO for WC2026 teams
    if seed_elo:
        for team, rating in seed_elo.items():
            updated[team] = rating

    def get(team):
        return updated.get(team, ELO_INITIAL)

    # Step 2: apply WC2026 results in chronological order
    for m in sorted(wc_matches, key=lambda x: x.get("date", "")):
        h, a = m["home"], m["away"]
        hg, ag = m["hg"], m["ag"]
        h_rating, a_rating = get(h), get(a)
        h_expected = _elo_expected(h_rating, a_rating)

        if hg > ag:
            h_actual, a_actual = 1.0, 0.0
        elif hg == ag:
            h_actual, a_actual = 0.5, 0.5
        else:
            h_actual, a_actual = 0.0, 1.0

        k = ELO_K * (1 + tournament_weight)
        updated[h] = h_rating + k * (h_actual - h_expected)
        updated[a] = a_rating + k * (a_actual - (1 - h_expected))

    return updated


def _apply_calibration(raw_probs: np.ndarray, calibrators) -> np.ndarray:
    """Apply per-class isotonic calibration and renormalise."""
    if calibrators is None:
        return raw_probs
    cal = np.column_stack([calibrators[c].predict(raw_probs[:, c]) for c in range(3)])
    row_sums = cal.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return cal / row_sums


def predict_match(home_team: str, away_team: str,
                  team_features: pd.DataFrame,
                  model=None, le=None, feature_cols=None, elo=None,
                  calibrators=None) -> dict:
    """
    Predict outcome probabilities for a single match.
    Returns {"home_win": float, "draw": float, "away_win": float}
    """
    if model is None:
        model, le, feature_cols, elo, calibrators = load_model()

    dummy_match = pd.DataFrame([{
        "home_team_name": home_team,
        "away_team_name": away_team,
        "home_score": np.nan,
        "away_score": np.nan,
        "tournament_name": "wc_2026",
        "neutral": True,
    }])

    match_df = build_match_features(dummy_match, team_features, elo)
    if match_df.empty:
        return {"home_win": 1/3, "draw": 1/3, "away_win": 1/3}

    X = match_df[[c for c in feature_cols if c in match_df.columns]] \
        .apply(pd.to_numeric, errors="coerce").fillna(0).values

    raw_probs = model.predict_proba(X)
    probs = _apply_calibration(raw_probs, calibrators)[0]
    classes = le.classes_

    return {cls: float(prob) for cls, prob in zip(classes, probs)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Use CUDA GPU for training")
    args = parser.parse_args()
    train(use_gpu=args.gpu)
