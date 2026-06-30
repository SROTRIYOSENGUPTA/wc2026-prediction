#!/usr/bin/env python3
"""Generate standalone publication figures for the WC2026 model article.
Outputs PNGs to article_charts/ — independent of the website.
Run: PYTHONPATH=. python3 make_article_charts.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

OUT = os.path.join(os.path.dirname(__file__), "article_charts")
os.makedirs(OUT, exist_ok=True)

# ---- house style ----------------------------------------------------------
GOLD, TEAL, GRAY, DARK = "#E0A21A", "#1B9E8F", "#B7BCC2", "#222428"
RED, BLUE, INK = "#D1495B", "#3A6EA5", "#2A2D34"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11.5,
    "axes.edgecolor": "#cfd3d8", "axes.linewidth": 0.9,
    "axes.titlesize": 15, "axes.titleweight": "bold",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": "#5b6068", "ytick.color": INK,
    "figure.dpi": 200, "savefig.dpi": 200,
})

def finish(fig, ax, title, sub, fname, src="Model: XGBoost (10 features) + 500k Monte Carlo  ·  S. Sengupta, WC 2026"):
    ax.set_title("")
    fig.text(0.012, 0.965, title, fontsize=16, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.012, 0.915, sub, fontsize=11, color="#6b7178", ha="left", va="top")
    fig.text(0.012, 0.022, src, fontsize=8.5, color="#9aa0a6", ha="left", va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", fname)

# ---- load data ------------------------------------------------------------
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "outputs/live_simulation_500k.csv"))

# real feature importances (from trained model)
FEATS = [("Ballon d'Or talent", 0.1356), ("Club chemistry", 0.1277), ("ELO rating", 0.1275),
         ("Net xG", 0.1185), ("Club prestige", 0.1184), ("Intl. experience", 0.1050),
         ("Club avg xG", 0.0966), ("Attack quality", 0.0908), ("Squad disruption", 0.0799),
         ("Knockout flag", 0.0)]

SEEDED_ELO = {"Spain":2171,"Argentina":2113,"France":2063,"England":2042,"Colombia":1998,
              "Brazil":1979,"Portugal":1976,"Netherlands":1959,"Croatia":1933,"Ecuador":1933,
              "Norway":1922,"Germany":1910,"Switzerland":1897,"Uruguay":1890}

# odds snapshots for the live-update story
SNAP_MID = {"France":15.21,"Argentina":16.94,"Spain":10.20,"England":5.18,"Netherlands":11.24,
            "Brazil":4.44,"Portugal":3.77,"Germany":2.05,"Belgium":4.09,"Croatia":5.17,"Colombia":4.94}
SNAP_FIN = dict(zip(df.team, df.win_pct))

# corners (verified game-by-game scrape): team -> (total, games)
CORN = {"Canada":(35,3),"Uruguay":(26,3),"England":(24,3),"Spain":(23,3),"Turkey":(22,3),
        "Senegal":(20,3),"Switzerland":(19,3),"USA":(19,3),"Egypt":(18,3),"Germany":(18,3),
        "Ecuador":(17,3),"Sweden":(17,3),"Brazil":(16,3),"Morocco":(16,3),"France":(15,3),
        "Algeria":(12,2),"Netherlands":(13,3)}

# ===========================================================================
# FIG 1 — Title odds
# ===========================================================================
top = df.head(14).iloc[::-1]
fig, ax = plt.subplots(figsize=(8.6, 6.4)); fig.subplots_adjust(top=0.84, left=0.20, bottom=0.155)
colors = [GOLD if t=="France" else (TEAL if t in ("Argentina","Spain","England") else GRAY) for t in top.team]
bars = ax.barh(top.team, top.win_pct, color=colors, edgecolor="white", height=0.74)
for b, v in zip(bars, top.win_pct):
    ax.text(b.get_width()+0.3, b.get_y()+b.get_height()/2, f"{v:.1f}%", va="center", fontsize=10, fontweight="bold", color=INK)
ax.set_xlim(0, max(top.win_pct)*1.16); ax.set_xlabel("Probability of winning the World Cup")
ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="#eef0f2", zorder=0)
ax.set_axisbelow(True)
finish(fig, ax, "Who wins the 2026 World Cup?",
       "500,000 simulated tournaments, group stage complete. France lead after a dominant group + favourable draw.",
       "fig1_title_odds.png")

# ===========================================================================
# FIG 2 — Road to the title (reach SF / Final / Win) for top 8
# ===========================================================================
t8 = df.head(8).iloc[::-1]
y = np.arange(len(t8)); h = 0.26
fig, ax = plt.subplots(figsize=(9.0, 6.2)); fig.subplots_adjust(top=0.84, left=0.16, bottom=0.155)
ax.barh(y+h, t8.semi_pct,  height=h, color="#cfe6e2", label="Reach Semis", edgecolor="white")
ax.barh(y,    t8.final_pct, height=h, color=TEAL,      label="Reach Final", edgecolor="white")
ax.barh(y-h, t8.win_pct,   height=h, color=GOLD,      label="Win title",   edgecolor="white")
for yy, r in zip(y, t8.itertuples()):
    ax.text(r.semi_pct+0.6, yy+h, f"{r.semi_pct:.0f}", va="center", fontsize=8.5, color="#5b6068")
    ax.text(r.final_pct+0.6, yy,  f"{r.final_pct:.0f}", va="center", fontsize=8.5, color=TEAL)
    ax.text(r.win_pct+0.6, yy-h,  f"{r.win_pct:.0f}", va="center", fontsize=8.5, color="#b6800f", fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(t8.team); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Probability (%)"); ax.set_xlim(0, max(t8.semi_pct)*1.12)
ax.grid(axis="x", color="#eef0f2"); ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False, fontsize=10)
finish(fig, ax, "The road to the trophy",
       "How far each contender is projected to go — reaching the semis is common; converting that to a title is rare.",
       "fig2_progression.png")

# ===========================================================================
# FIG 3 — Feature importance
# ===========================================================================
labels = [f[0] for f in FEATS][::-1]; vals = [f[1]*100 for f in FEATS][::-1]
fig, ax = plt.subplots(figsize=(8.6, 6.0)); fig.subplots_adjust(top=0.84, left=0.26, bottom=0.155)
cols = [GOLD if v>=12.5 else (TEAL if v>=10 else GRAY) for v in vals]
bars = ax.barh(labels, vals, color=cols, edgecolor="white", height=0.72)
for b, v in zip(bars, vals):
    txt = f"{v:.1f}%" if v>0 else "0.0% (pruned signal)"
    ax.text(b.get_width()+0.15, b.get_y()+b.get_height()/2, txt, va="center", fontsize=9.5,
            color=("#9aa0a6" if v==0 else INK), fontweight=("normal" if v==0 else "bold"))
ax.set_xlim(0, 15.5); ax.set_xlabel("Relative importance in the model (%)")
ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="#eef0f2"); ax.set_axisbelow(True)
finish(fig, ax, "What the model actually weighs",
       "Gain-based feature importance. Talent (Ballon d'Or), club chemistry and ELO lead; no single factor dominates.",
       "fig3_feature_importance.png")

# ===========================================================================
# FIG 4 — Odds shift (dumbbell): mid-group vs group complete
# ===========================================================================
teams = ["France","England","Spain","Argentina","Netherlands","Portugal","Colombia","Croatia","Belgium"]
pairs = [(t, SNAP_MID.get(t,0), SNAP_FIN.get(t,0)) for t in teams]
pairs.sort(key=lambda x: x[2])
yy = np.arange(len(pairs))
fig, ax = plt.subplots(figsize=(8.8, 6.2)); fig.subplots_adjust(top=0.84, left=0.16, bottom=0.155)
for i,(t,a,b) in enumerate(pairs):
    up = b>=a
    ax.plot([a,b],[i,i], color=("#76c7bd" if up else "#e3a3ab"), lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(a,i, s=70, color=GRAY, zorder=2)
    ax.scatter(b,i, s=95, color=(TEAL if up else RED), zorder=3)
    ax.text(b+(0.5 if b>=a else -0.5), i, f"{b:.1f}%", va="center", ha=("left" if b>=a else "right"),
            fontsize=9, fontweight="bold", color=(TEAL if up else RED))
    ax.text(a-0.5 if a<b else a+0.5, i+0.0, f"{a:.0f}", va="center", ha=("right" if a<b else "left"),
            fontsize=8, color="#9aa0a6")
ax.set_yticks(yy); ax.set_yticklabels([p[0] for p in pairs]); ax.tick_params(axis="y", length=0)
ax.set_xlim(-3, 30); ax.set_xlabel("Title probability (%)")
ax.grid(axis="x", color="#eef0f2"); ax.set_axisbelow(True)
ax.scatter([],[],color=GRAY,s=70,label="Mid–group stage (Jun 26)")
ax.scatter([],[],color=TEAL,s=90,label="Group stage complete (Jun 27)")
ax.legend(loc="lower right", frameon=False, fontsize=10)
finish(fig, ax, "The forecast updates itself",
       "Title odds before vs after the final group games. France & England surged; Belgium & Croatia drew tougher paths.",
       "fig4_odds_shift.png")

# ===========================================================================
# FIG 5 — Corners per game (real scraped data)
# ===========================================================================
cpg = sorted(((t, tot/g) for t,(tot,g) in CORN.items()), key=lambda x:x[1])[-12:]
labels = [c[0] for c in cpg]; vals=[c[1] for c in cpg]
fig, ax = plt.subplots(figsize=(8.4, 6.0)); fig.subplots_adjust(top=0.84, left=0.18, bottom=0.155)
cols=[GOLD if v>=9 else (TEAL if v>=7 else GRAY) for v in vals]
bars=ax.barh(labels, vals, color=cols, edgecolor="white", height=0.72)
for b,v in zip(bars,vals):
    ax.text(b.get_width()+0.1, b.get_y()+b.get_height()/2, f"{v:.1f}", va="center", fontsize=9.5, fontweight="bold")
ax.set_xlim(0, max(vals)*1.14); ax.set_xlabel("Corners won per game (group stage)")
ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="#eef0f2"); ax.set_axisbelow(True)
finish(fig, ax, "Set-piece pressure: corners per game",
       "From a full game-by-game scrape of all 66 group matches. Canada's set-piece volume is in a tier of its own.",
       "fig5_corners.png", src="Source: FotMob match data, scraped per game  ·  S. Sengupta, WC 2026")

# ===========================================================================
# FIG 6 — Seeded ELO (model's starting point)
# ===========================================================================
items = sorted(SEEDED_ELO.items(), key=lambda x:x[1])
labels=[i[0] for i in items]; vals=[i[1] for i in items]
fig, ax = plt.subplots(figsize=(8.4, 6.0)); fig.subplots_adjust(top=0.84, left=0.18, bottom=0.155)
cols=[GOLD if v>=2042 else GRAY for v in vals]
bars=ax.barh(labels, vals, color=cols, edgecolor="white", height=0.72)
for b,v in zip(bars,vals):
    ax.text(b.get_width()-12, b.get_y()+b.get_height()/2, f"{v:.0f}", va="center", ha="right",
            fontsize=9, fontweight="bold", color="white")
ax.set_xlim(1850, 2200); ax.set_xlabel("Pre-tournament ELO rating")
ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="#eef0f2"); ax.set_axisbelow(True)
finish(fig, ax, "Where the model starts: seeded ELO",
       "Pre-tournament strength ratings that anchor the simulation before any 2026 result is played.",
       "fig6_seeded_elo.png")

print("\nAll figures written to:", OUT)
