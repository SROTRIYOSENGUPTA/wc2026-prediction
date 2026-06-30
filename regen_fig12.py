#!/usr/bin/env python3
"""Regenerate ONLY fig1 (title odds) and fig2 (progression) from the corrected
real-bracket 500k sim, without touching the separately-corrected fig3/7/8.
Run: PYTHONPATH=. python3 regen_fig12.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "article_charts")
GOLD, TEAL, GRAY, INK = "#E0A21A", "#1B9E8F", "#B7BCC2", "#2A2D34"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11.5,
    "axes.edgecolor": "#cfd3d8", "axes.linewidth": 0.9,
    "axes.titlesize": 15, "axes.titleweight": "bold",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": "#5b6068", "ytick.color": INK,
    "figure.dpi": 200, "savefig.dpi": 200,
})

def finish(fig, ax, title, sub, fname,
           src="Model: XGBoost (10 features) + 500k Monte Carlo  ·  S. Sengupta, WC 2026"):
    ax.set_title("")
    fig.text(0.012, 0.965, title, fontsize=16, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.012, 0.915, sub, fontsize=11, color="#6b7178", ha="left", va="top")
    fig.text(0.012, 0.022, src, fontsize=8.5, color="#9aa0a6", ha="left", va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", fname)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "outputs/live_simulation_500k.csv"))

# FIG 1 — Title odds
top = df.head(14).iloc[::-1]
fig, ax = plt.subplots(figsize=(8.6, 6.4)); fig.subplots_adjust(top=0.84, left=0.20, bottom=0.155)
colors = [GOLD if t == "France" else (TEAL if t in ("Argentina", "Spain", "England") else GRAY) for t in top.team]
bars = ax.barh(top.team, top.win_pct, color=colors, edgecolor="white", height=0.74)
for b, v in zip(bars, top.win_pct):
    ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
            va="center", fontsize=10, fontweight="bold", color=INK)
ax.set_xlim(0, max(top.win_pct) * 1.16); ax.set_xlabel("Probability of winning the World Cup")
ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="#eef0f2", zorder=0)
ax.set_axisbelow(True)
finish(fig, ax, "Who wins the 2026 World Cup?",
       "500,000 simulated tournaments over the real knockout draw (Germany eliminated). "
       "France lead the marginal odds; Spain is the most-likely single bracket.",
       "fig1_title_odds.png")

# FIG 2 — Road to the title (reach SF / Final / Win) for top 8
t8 = df.head(8).iloc[::-1]
y = np.arange(len(t8)); h = 0.26
fig, ax = plt.subplots(figsize=(9.0, 6.2)); fig.subplots_adjust(top=0.84, left=0.16, bottom=0.155)
ax.barh(y + h, t8.semi_pct,  height=h, color="#cfe6e2", label="Reach Semis", edgecolor="white")
ax.barh(y,     t8.final_pct, height=h, color=TEAL,      label="Reach Final", edgecolor="white")
ax.barh(y - h, t8.win_pct,   height=h, color=GOLD,      label="Win title",   edgecolor="white")
for yy, r in zip(y, t8.itertuples()):
    ax.text(r.semi_pct + 0.6, yy + h, f"{r.semi_pct:.0f}", va="center", fontsize=8.5, color="#5b6068")
    ax.text(r.final_pct + 0.6, yy,    f"{r.final_pct:.0f}", va="center", fontsize=8.5, color=TEAL)
    ax.text(r.win_pct + 0.6, yy - h,  f"{r.win_pct:.0f}", va="center", fontsize=8.5, color="#b6800f", fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(t8.team); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Probability (%)"); ax.set_xlim(0, max(t8.semi_pct) * 1.12)
ax.grid(axis="x", color="#eef0f2"); ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False, fontsize=10)
finish(fig, ax, "The road to the trophy",
       "How far each contender is projected to go — reaching the semis is common; converting that to a title is rare.",
       "fig2_progression.png")
