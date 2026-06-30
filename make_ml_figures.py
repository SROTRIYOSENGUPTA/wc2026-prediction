#!/usr/bin/env python3
"""Five technical ML figures for the article, all from real artifacts:
  1. SHAP summary (beeswarm)        — outputs/oof_eval.npz X + trained model
  2. Calibration curve              — OOF calibrated probabilities vs outcomes
  3. Log-loss convergence           — XGBoost train/valid mlogloss per boosting round
  4. Monte Carlo convergence        — running title-prob estimate vs # simulations
  5. Posterior title distribution   — Beta(k+1, N-k+1) posterior on each team's p(title)
Run: PYTHONPATH=. python3 make_ml_figures.py
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve
from scipy.stats import beta as beta_dist
import xgboost as xgb, shap

from src.model import load_model, WC2026_SEEDED_ELO, update_elo_with_live_results
from src.ingest.live_results import COMPLETED_MATCHES, COMPLETED_KNOCKOUT
from src.simulation import ProbabilityCache

OUT = os.path.join(os.path.dirname(__file__), "article_charts")
GOLD, TEAL, GRAY, INK = "#E0A21A", "#1B9E8F", "#B7BCC2", "#2A2D34"
RED, BLUE = "#D1495B", "#3A6EA5"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11.5,
    "axes.edgecolor": "#cfd3d8", "axes.linewidth": 0.9,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": "#5b6068", "ytick.color": INK,
    "figure.dpi": 200, "savefig.dpi": 200,
})
SRC = "Model: XGBoost (10 features) + 500k Monte Carlo  ·  S. Sengupta, WC 2026"

def cap(fig, title, sub):
    fig.text(0.012, 0.972, title, fontsize=16, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.012, 0.922, sub, fontsize=10.5, color="#6b7178", ha="left", va="top")
    fig.text(0.012, 0.02, SRC, fontsize=8.5, color="#9aa0a6", ha="left", va="bottom")

def save(fig, fname):
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("wrote", fname)

# ---- load cached training artifacts + model -------------------------------
d = np.load(os.path.join(os.path.dirname(__file__), "outputs/oof_eval.npz"), allow_pickle=True)
X, y, oof, cal, feats = d["X"], d["y"], d["oof"], d["cal"], list(d["feats"])
PRETTY = {"elo_diff":"ELO diff","net_xg_diff":"Net xG diff","squad_avg_club_xg_diff":"Club xG diff",
          "squad_attack_quality_diff":"Attack quality diff","squad_club_prestige_diff":"Club prestige diff",
          "squad_club_chemistry_diff":"Chemistry diff","avg_intl_matches_diff":"Intl. experience diff",
          "pct_significant_drop_diff":"Squad disruption diff","is_knockout":"Knockout flag",
          "squad_ballon_dor_diff":"Ballon d'Or diff"}
fnames = [PRETTY.get(f, f) for f in feats]
model, le, fc, elo_base, calibrators = load_model()
classes = list(le.classes_)
home_idx = next((i for i, c in enumerate(classes) if "home" in str(c).lower() or str(c) in ("2", "H")), len(classes) - 1)

# ===========================================================================
# 1. SHAP summary (beeswarm) for the home-win class
# ===========================================================================
expl = shap.TreeExplainer(model)
sv = expl.shap_values(X)
if isinstance(sv, list):
    arr = sv[home_idx]
else:
    sv = np.asarray(sv); arr = sv[:, :, home_idx] if sv.ndim == 3 else sv
plt.figure(figsize=(8.8, 6.2))
shap.summary_plot(arr, X, feature_names=fnames, show=False, plot_size=None, color_bar=True, max_display=10)
fig = plt.gcf(); fig.set_size_inches(8.8, 6.6); fig.subplots_adjust(top=0.82, left=0.30, bottom=0.12)
ax = plt.gca(); ax.set_xlabel("SHAP value  →  pushes toward a HOME win", fontsize=10.5)
cap(fig, "What drives a prediction (SHAP)",
    "Per-match feature contributions for the home-win class. Red = high feature value, blue = low; "
    "right = pushes the model toward the stronger side winning.")
save(fig, "fig_shap_summary.png")

# ===========================================================================
# 2. Calibration curve (reliability diagram) — calibrated OOF, per class
# ===========================================================================
fig, ax = plt.subplots(figsize=(7.4, 6.6)); fig.subplots_adjust(top=0.82, left=0.13, bottom=0.12)
ax.plot([0, 1], [0, 1], "--", color="#9aa0a6", lw=1.2, label="Perfect calibration")
cols = {0: BLUE, 1: GRAY, 2: GOLD}
order = sorted(range(len(classes)), key=lambda i: (i != home_idx))
for ci in range(len(classes)):
    frac, mean = calibration_curve((y == ci).astype(int), cal[:, ci], n_bins=8, strategy="quantile")
    ax.plot(mean, frac, "o-", color=cols.get(ci, TEAL), lw=2, ms=6, label=f"{classes[ci]}")
ll_cal = log_loss(y, cal); ll_oof = log_loss(y, oof)
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.grid(color="#eef0f2"); ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.text(0.97, 0.06, f"OOF log-loss (calibrated): {ll_cal:.3f}\nOOF log-loss (raw): {ll_oof:.3f}\nrandom baseline: {np.log(3):.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color="#5b6068",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f6f7f8", ec="#e2e5e8"))
for s in ("top", "right"): ax.spines[s].set_visible(False)
cap(fig, "Is the model honest? (calibration)",
    "Out-of-fold predicted probability vs how often it actually happened. On the diagonal = well-calibrated.")
save(fig, "fig_calibration_curve.png")

# ===========================================================================
# 3. Log-loss convergence — train vs validation mlogloss per boosting round
# ===========================================================================
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=4,
                        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                        random_state=42)
clf.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
ev = clf.evals_result(); rounds = np.arange(1, len(ev["validation_0"]["mlogloss"]) + 1)
fig, ax = plt.subplots(figsize=(8.6, 6.0)); fig.subplots_adjust(top=0.82, left=0.12, bottom=0.13)
ax.plot(rounds, ev["validation_0"]["mlogloss"], color=TEAL, lw=2, label="Train")
ax.plot(rounds, ev["validation_1"]["mlogloss"], color=RED, lw=2, label="Validation (held-out)")
ax.axhline(np.log(3), ls="--", color="#9aa0a6", lw=1.1, label=f"Random baseline (ln 3 = {np.log(3):.3f})")
ax.axvline(500, ls=":", color="#b6800f", lw=1.2)
ax.text(500, ax.get_ylim()[1]*0.97, " 500 trees (chosen)", color="#b6800f", fontsize=9, va="top")
ax.set_xlabel("Boosting rounds (trees)"); ax.set_ylabel("Multiclass log-loss")
ax.grid(color="#eef0f2"); ax.set_axisbelow(True); ax.legend(loc="upper right", frameon=False, fontsize=10)
for s in ("top", "right"): ax.spines[s].set_visible(False)
cap(fig, "When does learning stop? (log-loss convergence)",
    "Training keeps dropping; validation flattens. The gap is the regularization (L2=5, min_child_weight=4) holding overfit in check.")
save(fig, "fig_logloss_convergence.png")

# ===========================================================================
# Run the real-bracket 500k sim ONCE (shared by figs 4 & 5)
# ===========================================================================
N = 500_000; np.random.seed(42)
R32 = [("Germany","Paraguay"),("France","Sweden"),("South Africa","Canada"),("Netherlands","Morocco"),
       ("Portugal","Croatia"),("Spain","Austria"),("USA","Bosnia and Herzegovina"),("Belgium","Senegal"),
       ("Brazil","Japan"),("Ivory Coast","Norway"),("Mexico","Ecuador"),("England","DR Congo"),
       ("Argentina","Cabo Verde"),("Australia","Egypt"),("Switzerland","Algeria"),("Colombia","Ghana")]
tf = pd.read_parquet("data/processed/team_features.parquet")
elo = update_elo_with_live_results(elo_base, COMPLETED_MATCHES + COMPLETED_KNOCKOUT,
                                   seed_elo=WC2026_SEEDED_ELO, tournament_weight=1.0)
pc = ProbabilityCache(tf, model, le, fc, elo, calibrators)
teams = [t for pr in R32 for t in pr]; idx = {t: i for i, t in enumerate(teams)}
ko_winner = {frozenset((m["home"], m["away"])): m["winner"] for m in COMPLETED_KNOCKOUT}
padv = np.full((32, 32), 0.5)
for i, a in enumerate(teams):
    for j, b in enumerate(teams):
        if i != j:
            ph, pd_, pa = pc.get(a, b); padv[i, j] = ph + pd_ / 2.0
def resolve(aa, bb):
    return np.where(np.random.random(len(aa)) < padv[aa, bb], aa, bb).astype(np.int32)
r32w = []
for h, a in R32:
    fr = ko_winner.get(frozenset((h, a)))
    if fr is not None: r32w.append(np.full(N, idx[fr], dtype=np.int32))
    else: r32w.append(np.where(np.random.random(N) < padv[idx[h], idx[a]], idx[h], idx[a]).astype(np.int32))
def nextr(ws): return [resolve(ws[k], ws[k+1]) for k in range(0, len(ws), 2)]
champ = resolve(*[np.concatenate([x]) for x in [None]][:0] or [None]) if False else None
r16 = nextr(r32w); qf = nextr(r16); sf = nextr(qf)
champ = resolve(sf[0], sf[1])
counts = np.bincount(champ, minlength=32)
top = sorted(range(32), key=lambda i: -counts[i])[:6]
COLR = {0: GOLD, 1: TEAL, 2: BLUE, 3: RED, 4: "#7B68EE", 5: "#5b6068"}

# ===========================================================================
# 4. Monte Carlo convergence — running title-prob estimate vs N
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.8, 6.0)); fig.subplots_adjust(top=0.82, left=0.11, bottom=0.13)
xs = np.unique(np.geomspace(50, N, 400).astype(int))
for rank, ti in enumerate(top[:5]):
    run = np.cumsum(champ == ti) / np.arange(1, N + 1)
    ax.plot(xs, run[xs - 1] * 100, color=COLR[rank], lw=1.8, label=f"{teams[ti]} ({counts[ti]/N*100:.1f}%)")
ax.set_xscale("log"); ax.set_xlabel("Number of simulated tournaments"); ax.set_ylabel("Estimated title probability (%)")
ax.grid(color="#eef0f2"); ax.set_axisbelow(True); ax.legend(loc="upper right", frameon=False, fontsize=9.5, ncol=2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
cap(fig, "How many simulations are enough? (Monte Carlo convergence)",
    "Running title-probability estimate as simulations accumulate. Estimates are noisy early and lock in by ~100k runs.")
save(fig, "fig_mc_convergence.png")

# ===========================================================================
# 5. Posterior title distribution — Beta(k+1, N-k+1) per team, uniform prior
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.8, 6.0)); fig.subplots_adjust(top=0.82, left=0.10, bottom=0.13)
lo = max(0, (min(counts[t] for t in top) / N) - 0.02); hi = (max(counts[t] for t in top) / N) + 0.025
xx = np.linspace(lo, hi, 2000)
for rank, ti in enumerate(top):
    k = counts[ti]; dens = beta_dist.pdf(xx, k + 1, N - k + 1)
    ax.plot(xx * 100, dens, color=COLR[rank], lw=2, label=f"{teams[ti]}")
    ax.fill_between(xx * 100, dens, color=COLR[rank], alpha=0.10)
ax.set_xlabel("Title probability (%)"); ax.set_ylabel("Posterior density")
ax.set_yticks([]); ax.grid(axis="x", color="#eef0f2"); ax.set_axisbelow(True)
ax.legend(loc="upper right", frameon=False, fontsize=10)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
cap(fig, "How sure are the odds? (posterior over title probability)",
    "Beta posterior on each contender's true title probability given 500k simulations (uniform prior). "
    "Non-overlap = a statistically real ordering, not Monte Carlo noise.")
save(fig, "fig_posterior_title.png")
print("\nAll 5 figures written to article_charts/")
