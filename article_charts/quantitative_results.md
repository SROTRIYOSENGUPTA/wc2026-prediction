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
| Log-loss (calibrated, OOF) | **0.942** | base-rate 1.065 |
| Log-loss (raw, OOF) | 1.032 | uniform 1.099 |
| Top-pick accuracy (OOF) | **54.7%** | majority-class 43.7% |
| In-sample calibrated (do **not** quote as performance) | 0.78 | — |

_A real, modest edge: ~12% lower log-loss than the base rate, +11 pts accuracy over always-picking-the-favourite. The earlier "0.44" was in-sample, post-calibration on the buggy 666-row fit — optimistic and not reproducible._

## Feature importance (corrected 980-match model)
| Feature | Importance |
|---|--:|
| Net xG | 12.8% |
| ELO rating | 12.2% |
| Club prestige | 11.4% |
| Ballon d'Or talent | 10.9% |
| Intl. experience | 10.5% |
| Club chemistry | 9.7% |
| Squad disruption | 8.6% |
| Club avg xG | 8.4% |
| Attack quality | 8.2% |
| **Knockout flag** | **7.2%** (was 0% — the dropped tournament matches gave it variance) |

## Title odds (corrected 980-match + in-tournament-form model, real knockout draw, 500k sim)
_Re-run with **14 of 16 R32 results locked** (all model-correct): Germany out to Paraguay & Netherlands out to Morocco on pens; France 3–0, Spain 3–0, Portugal 2–1, USA 2–0, England 2–1, Mexico 2–0, Switzerland 2–0, Belgium (aet), Norway, Egypt (pens). Argentina–Cabo Verde and Colombia–Ghana still to play._
France 25.6 · Spain 20.0 · Argentina 14.0 · England 8.4 · Brazil 8.2 · Portugal 4.3 · Norway 3.4 · Morocco 2.8 …
- **Modal (chalk) bracket:** Spain champion over Argentina (66.5% Final); Spain beats France in the SF (59.1%); Brazil edges Norway then the hosts to reach the semis before Argentina ends them; host Mexico ousts England in the R16.
- **Marginal (Monte Carlo):** France remain the most-likely single winner across all simulated worlds despite Spain winning the single most-likely bracket.

## Figure manifest
- fig1 title odds · fig2 progression — **corrected (980 + form, real-bracket 500k sim)**
- fig3 feature importance — **corrected (980)**
- fig4 odds shift — **conflated metric; drop from article**
- fig5 corners (scraped data, unaffected) · fig6 seeded ELO (unaffected)
- fig7 calibration — **corrected (980), OOF, cal. log-loss 0.94**
- fig8 feature correlation — **corrected (980), knockout flag included**
