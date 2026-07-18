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

## Feature importance (retrained; median-imputed + depth-aware Ballon d'Or)
_Ballon d'Or squad score is now a geometric-decay weighted sum over the whole squad's 2025 rankings (rewards depth: France's Dembélé #1 + Mbappé #5), not just the single best player._

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
_Every knockout through the semifinals is locked. Both SFs were model coin flips and both went chalk: Spain 2–0 France, Argentina 2–1 England. The real Final — Spain vs Argentina — is exactly the Final the model projected from the group stage. Deterministic bracket now seeds the same pre-tournament ELO the sim uses, so the two agree. Missing squad features imputed with the cross-team median (fixes a UEFA-favouring bias from zero-fill)._
Spain 66.2 · Argentina 33.8 (Final is set)
- **Modal (chalk) bracket:** Spain champion over Argentina (69.2% head-to-head; 66.2% incl. pens paths); the coin-flip SF over France (52.2%) went Spain's way for real, 2–0.
- **Marginal (Monte Carlo):** with the field down to the two finalists, modal and marginal coincide: Spain 66.2%.
- Remaining: only the Final (Spain 66.2/33.8). England took bronze 5–3 over France — an upset (model: France 79.8%).
- **Individual hot-hand:** ±5% post-prediction boost toward the team with the hotter in-tournament scorer (goals + ½·assists) — France (Mbappé 9.0), Argentina (Messi 8.0), Norway (Haaland 7.0) benefit.

## Figure manifest
- fig1 title odds · fig2 progression — **corrected (980 + form, real-bracket 500k sim)**
- fig3 feature importance — **corrected (980)**
- fig4 odds shift — **conflated metric; drop from article**
- fig5 corners (scraped data, unaffected) · fig6 seeded ELO (unaffected)
- fig7 calibration — **corrected (980), OOF, cal. log-loss 0.94**
- fig8 feature correlation — **corrected (980), knockout flag included**
