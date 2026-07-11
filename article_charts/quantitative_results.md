# WC 2026 Model — Quantitative Results (corrected, for the article)

_Model retrained after fixing a name-column bug that had been silently dropping 314 matches.
Training set is now the full **980 internationals**. Snapshot: group stage complete (Jun 27, 2026)._

## Training data (now accurate)
- **980 international matches.** Composition: **314 StatsBomb major-tournament games** — World Cup 128, Euro 102, AFCON 52, Copa América 32 — plus **666 additional international results** (team names + scores).
- Each match weighted by tournament importance for ELO (WC 1.0 → friendly 0.15). Weights drive ELO + form features, **not** the XGBoost loss.
- Labels: 3-class (home win 33.7% / draw 22.7% / away win 43.7%).

## Model spec
| | |
|---|---|
| Algorithm | XGBoost `multi:softprob`, 3 classes |
| Trees / depth / lr | 500 / 5 / 0.05 |
| Subsample / colsample | 0.8 / 0.8 |
| Regularization | `reg_lambda=5`, `min_child_weight=4` |
| Features | 10 squad-level home−away differences |
| Calibration | 5-fold OOF isotonic, per class, renormalized |

## Honest performance (out-of-fold, 5-fold CV) — use these, not 0.44
| Metric | Model | Baseline |
|---|--:|--:|
| Log-loss (calibrated, OOF) | **0.943** | base-rate 1.065 |
| Log-loss (raw, OOF) | 1.036 | uniform 1.099 |
| Top-pick accuracy (OOF) | **55.2%** | majority-class 43.7% |
| In-sample calibrated (do **not** quote as performance) | 0.78 | — |

_A real, modest edge: ~11% lower log-loss than the base rate, +11 pts accuracy over always-picking-the-favourite. Model **retrained with median imputation** for missing squad features (was zero-fill, which systematically underrated ~14 non-UEFA teams). Train and inference now use the same imputed features; OOF performance held (0.943, 55.2%)._

## Feature importance (retrained, median-imputed 980-match model)
| Feature | Importance |
|---|--:|
| Ballon d'Or talent | 12.6% |
| ELO rating | 12.2% |
| Club prestige | 11.5% |
| Intl. experience | 10.2% |
| Net xG | 10.0% |
| Club chemistry | 9.9% |
| Squad disruption | 9.0% |
| Attack quality | 8.8% |
| Club avg xG | 8.8% |
| **Knockout flag** | **7.1%** |

## Title odds (retrained + median-imputed model, real knockout draw, 500k sim)
_All 16 R32, all 8 R16, and the first two QFs locked (France 2–0 Morocco, Spain 2–1 Belgium). Six teams remain. Deterministic bracket now seeds the same pre-tournament ELO the sim uses, so the two agree. Missing squad features imputed with the cross-team median (fixes a UEFA-favouring bias from zero-fill)._
Spain 40.0 · France 34.8 · Argentina 13.0 · England 8.1 · Norway 3.2 · Switzerland 0.9
- **Modal (chalk) bracket:** Spain champion over Argentina (73.6% Final); Spain edges France in the SF (56.5%). Both are locked into that semifinal, so it effectively decides the title.
- **Marginal (Monte Carlo):** Spain is now both the chalk champion and the marginal favourite (40.0%); France 2nd (34.8%). The modal-vs-marginal gap has closed as the field thinned to six.
- **Coin flip** (unplayed tie, favourite ≤55%): Norway–England (QF).

## Figure manifest
- fig1 title odds · fig2 progression — **corrected (980 + form, real-bracket 500k sim)**
- fig3 feature importance — **corrected (980)**
- fig4 odds shift — **conflated metric; drop from article**
- fig5 corners (scraped data, unaffected) · fig6 seeded ELO (unaffected)
- fig7 calibration — **corrected (980), OOF, cal. log-loss 0.94**
- fig8 feature correlation — **corrected (980), knockout flag included**
