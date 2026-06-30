#!/usr/bin/env python3
"""500k Monte Carlo over the REAL confirmed knockout bracket (group stage complete).

Bypasses the generic group-slot allocation in simulate_live_tournament — the draw
is now known, so we seed the actual R32 matchups and simulate forward, LOCKING any
completed knockout results (actual winner advances with probability 1).
Uses ProbabilityCache so the host-familiarity and in-tournament-form adjustments apply.
Writes outputs/live_simulation_500k.csv (same schema the figures read).
"""
import numpy as np, pandas as pd
from pathlib import Path
from src.model import load_model, WC2026_SEEDED_ELO, update_elo_with_live_results
from src.ingest.live_results import COMPLETED_MATCHES, COMPLETED_KNOCKOUT
from src.simulation import ProbabilityCache

N = 500_000
np.random.seed(42)

# Real confirmed R32 bracket (FotMob), in bracket order: first 8 = LEFT half, last 8 = RIGHT.
# Adjacent pairs feed each R16; LEFT and RIGHT only meet in the Final.
R32 = [("Germany","Paraguay"),("France","Sweden"),("South Africa","Canada"),("Netherlands","Morocco"),
       ("Portugal","Croatia"),("Spain","Austria"),("USA","Bosnia and Herzegovina"),("Belgium","Senegal"),
       ("Brazil","Japan"),("Ivory Coast","Norway"),("Mexico","Ecuador"),("England","DR Congo"),
       ("Argentina","Cabo Verde"),("Australia","Egypt"),("Switzerland","Algeria"),("Colombia","Ghana")]

model, le, fc, elo_base, cal = load_model()
tf = pd.read_parquet("data/processed/team_features.parquet")
elo = update_elo_with_live_results(elo_base, COMPLETED_MATCHES + COMPLETED_KNOCKOUT,
                                   seed_elo=WC2026_SEEDED_ELO, tournament_weight=1.0)
pc = ProbabilityCache(tf, model, le, fc, elo, cal)

teams = [t for pair in R32 for t in pair]          # 32 distinct knockout teams
idx = {t: i for i, t in enumerate(teams)}
ko_winner = {frozenset((m["home"], m["away"])): m["winner"] for m in COMPLETED_KNOCKOUT}

# Precompute advance-probabilities padv[i][j] = P(team i beats team j) incl. ET/pens split
padv = np.full((32, 32), 0.5)
for i, a in enumerate(teams):
    for j, b in enumerate(teams):
        if i == j: continue
        ph, pd_, pa = pc.get(a, b)
        padv[i, j] = ph + pd_ / 2.0

def resolve(a_arr, b_arr):
    p = padv[a_arr, b_arr]
    return np.where(np.random.random(len(a_arr)) < p, a_arr, b_arr).astype(np.int32)

# R32 (fixed pairs; locked where completed)
r32w = []
for h, a in R32:
    fr = ko_winner.get(frozenset((h, a)))
    if fr is not None:
        r32w.append(np.full(N, idx[fr], dtype=np.int32))
    else:
        p = padv[idx[h], idx[a]]
        r32w.append(np.where(np.random.random(N) < p, idx[h], idx[a]).astype(np.int32))

def nextround(ws):
    return [resolve(ws[k], ws[k+1]) for k in range(0, len(ws), 2)]

r16 = nextround(r32w)        # 8 (QF participants)
qf  = nextround(r16)         # 4 (SF participants)
sf  = nextround(qf)          # 2 (finalists)
champ = resolve(sf[0], sf[1])

def pct(arrs):
    c = np.zeros(32)
    for arr in arrs: c += np.bincount(arr, minlength=32)
    return c / N * 100

win  = np.bincount(champ, minlength=32) / N * 100
fin  = pct(sf)      # reached final
semi = pct(qf)      # reached semis
qfp  = pct(r16)     # reached QF
r16p = pct(r32w)    # reached R16

df = pd.DataFrame({"team": teams, "win_pct": win, "final_pct": fin,
                   "semi_pct": semi, "qf_pct": qfp, "r16_pct": r16p}).sort_values("win_pct", ascending=False)
df = df.round(2)
df.to_csv("outputs/live_simulation_500k.csv", index=False)
print("Real-bracket sim complete. Germany locked out (lost R32 to Paraguay on pens).\n")
print(df.head(14).to_string(index=False))
print("\nGermany title odds (should be 0):", round(float(win[idx["Germany"]]), 3), "%")
print("Paraguay reaches R16 in", round(float(r16p[idx["Paraguay"]]), 1), "% (locked through)")
