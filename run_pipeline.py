"""
Master pipeline runner.

Usage:
    python run_pipeline.py                         # full data pipeline
    python run_pipeline.py --skip-club             # skip slow FBref pull
    python run_pipeline.py --features-only         # rebuild features from cache
    python run_pipeline.py --train                 # train match model after features
    python run_pipeline.py --train --gpu           # train on GPU (Amaral cluster)
    python run_pipeline.py --simulate              # run 100k tournament simulations
    python run_pipeline.py --simulate --gpu        # run simulations on GPU
    python run_pipeline.py --simulate --n 500000   # 500k simulations
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ingest.squads import build_squad_master
from src.ingest.international import pull_all_international
from src.ingest.club_seasons import pull_all_club_seasons
from src.ingest.coach_history import build_coach_history_table
from src.features import build_all_features


def main(skip_club: bool = False, features_only: bool = False,
         train: bool = False, simulate: bool = False,
         live: bool = False,
         use_gpu: bool = False, n_sims: int = 100_000):

    print("=" * 60)
    print("  WC2026 Prediction Pipeline")
    print("=" * 60)

    if not features_only:
        print("\n[1/4] Pulling WC2026 squads...")
        squads = build_squad_master()
        print(f"      → {len(squads)} players loaded")

        print("\n[2/4] Pulling international data (StatsBomb + Nations League)...")
        matches, events = pull_all_international()
        print(f"      → {len(matches)} matches, {len(events)} with player-level events")

        if not skip_club:
            print("\n[3/4] Pulling club season data (FBref 2021-2026)...")
            print("      Note: ~45 minutes due to rate limits. Cached per file.")
            club = pull_all_club_seasons()
            print(f"      → {len(club)} player-season rows")
        else:
            print("\n[3/4] Skipping club season pull (--skip-club)")

        print("\n[4/4] Building coach-player history (Transfermarkt)...")
        coach = build_coach_history_table()
        print(f"      → {len(coach)} player-coach rows")

    print("\n[5/5] Building feature vectors...")
    player_features, team_features = build_all_features()
    print(f"      → {len(player_features)} players, {len(team_features)} teams")

    if train:
        print("\n[6] Training match outcome model...")
        print(f"    Device: {'GPU (CUDA)' if use_gpu else 'CPU'}")
        from src.model import train as train_model
        train_model(use_gpu=use_gpu)

    if simulate:
        print(f"\n[7] Running {n_sims:,} tournament simulations...")
        print(f"    Device: {'GPU (CuPy)' if use_gpu else 'CPU (NumPy)'}")
        if use_gpu:
            print("    Install CuPy on cluster: pip install cupy-cuda12x")
        from src.simulation import simulate_tournament
        simulate_tournament(n_sims=n_sims, use_gpu=use_gpu,
                            team_features=team_features)

    if live:
        print(f"\n[7] Running {n_sims:,} LIVE simulations from current WC2026 state...")
        print(f"    Device: {'GPU (CuPy)' if use_gpu else 'CPU (NumPy)'}")
        from src.model import load_model, update_elo_with_live_results, MODEL_DIR, WC2026_SEEDED_ELO
        from src.simulation import simulate_live_tournament
        from src.features import TEAM_NAME_NORMALIZE
        from src.ingest.live_results import COMPLETED_MATCHES
        model_path = MODEL_DIR / "match_model.pkl"
        if model_path.exists():
            model, le, feature_cols, elo, calibrators = load_model()
            # Seed with real-world ELO from worldcupelo.com, then apply live WC2026 results
            elo = update_elo_with_live_results(
                elo, COMPLETED_MATCHES,
                name_map=TEAM_NAME_NORMALIZE,
                tournament_weight=1.0,
                seed_elo=WC2026_SEEDED_ELO,
            )
            print(f"    ELO seeded from worldcupelo.com + {len(COMPLETED_MATCHES)} WC2026 matches applied")
        else:
            model, le, feature_cols, elo, calibrators = None, None, None, None, None
        simulate_live_tournament(
            n_sims=n_sims, use_gpu=use_gpu,
            team_features=team_features,
            model=model, le=le, feature_cols=feature_cols,
            elo=elo, calibrators=calibrators,
        )

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-club",     action="store_true")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--train",         action="store_true",
                        help="Train XGBoost match model after feature build")
    parser.add_argument("--simulate",      action="store_true",
                        help="Run Monte Carlo tournament simulation")
    parser.add_argument("--live",          action="store_true",
                        help="Simulate forward from current live WC2026 state")
    parser.add_argument("--gpu",           action="store_true",
                        help="Use GPU for training/simulation (Amaral cluster)")
    parser.add_argument("--n",             type=int, default=100_000,
                        help="Number of simulations (default: 100000)")
    args = parser.parse_args()
    main(skip_club=args.skip_club, features_only=args.features_only,
         train=args.train, simulate=args.simulate, live=args.live,
         use_gpu=args.gpu, n_sims=args.n)
