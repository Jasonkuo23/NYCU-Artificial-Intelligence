# MLB Bet365 Odds -> Profit-Optimized ML (SportsbookReview)

This repo does two things:

1) **Collect MLB odds** (Moneyline + Point Spread) from SportsbookReview (SBR) for **Bet365** and convert them to **Decimal odds**.
2) **Train supervised models** that are evaluated by **betting profit/ROI** (not just accuracy).

---

## 0) Setup

Create a venv (if you don’t have one):

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

---

## 1) Data you already have

You currently have yearly CSVs:

- `2021.csv`
- `2022.csv`, `2023.csv`, `2024.csv`, `2025.csv`

Schema (same as crawler output):

- `date` (YYYY-MM-DD)
- `game_id` (SBR matchup id)
- `away_team`, `home_team` (SBR `shortName` codes)
- `away_score`, `home_score`
- `result` (`home_win` or `away_win`)
- `spread_result` (`home_cover`, `away_cover`, or `push`)
- `moneyline_away_decimal`, `moneyline_home_decimal`
- `away_spread`, `home_spread`
- `spread_away_decimal`, `spread_home_decimal`

These files are already labeled for training (scores and outcomes are included directly by the crawler).

---

## 2) (Optional) Re-scrape odds from SBR by date range

Script: `crawl_odds.py`

Example: scrape a single day:

```bash
.venv/bin/python crawl_odds.py --start-date 2025-03-18 --end-date 2025-03-18 --output sbr_mlb_odds.csv
```

Notes:
- It fetches 2 SBR pages per day (moneyline + point spread).
- It filters to **Bet365** only.
- It outputs **Decimal odds** (converted from American odds in SBR JSON).
- It also outputs `away_score`, `home_score`, `result`, and `spread_result`.

---

## 3) Prepare one training file

`ml/run_experiments.py` expects one input CSV. If you have separate yearly files, merge them once:

```bash
.venv/bin/python - <<'PY'
import pandas as pd

files = ["2021.csv", "2022.csv", "2023.csv", "2024.csv", "2025.csv"]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv("all_2021_2025.csv", index=False)
print("rows", len(df), "-> all_2021_2025.csv")
PY
```

---

## 4) Train + evaluate for profit

Script: `ml/train_profit.py`

It trains a classifier (LR/RF/XGB) and evaluates:
- Standard metrics (logloss/AUC/accuracy)
- **Profit + ROI** using an EV-based betting policy

### 4.1 Moneyline model

```bash
.venv/bin/python ml/train_profit.py \
  --data all_2021_2025.csv \
  --market moneyline \
  --model lr \
  --test-start-date 2025-01-01 \
  --cv-folds 5 \
  --ev-threshold 0.0 \
  --out-predictions preds_2025_moneyline.csv
```

For report-level experiments (model comparison, CV, confusion matrices, class balancing, augmentation, PCA), use [ml/run_experiments.py](ml/run_experiments.py) in Section 9.

### What the result looks like

Console output looks like this (example):

- `CV logloss (train, 3 folds): mean=... std=...`
- `Test rows: ... (from YYYY-MM-DD)`
- `LogLoss: ...  AUC: ...  Acc: ...`
- `Strategy (model): bets=... profit=... ROI=...%`

The prediction file (e.g. `preds_2025_moneyline.csv`) contains per-game probabilities and the realized per-game profit under the policy:

- `p_home` (moneyline) or `p_home_cover` (spread)
- `y_true`
- `odds_home`, `odds_away`
- `ev_home`, `ev_away`
- `pick_side` (`home` or `away`)
- `bet` (1 if EV threshold passed, else 0)
- `profit` (profit for $1 stake; 0.0 typically means “no bet”)

---

## 5) How betting profit is computed

For each game the model outputs a probability for the **home** side:

- Moneyline: $p = P(\text{home wins})$
- Spread: $p = P(\text{home covers})$

Given decimal odds $o_{home}$ and $o_{away}$, expected value (EV) for a $1 bet is:

- $EV_{home} = p \cdot o_{home} - 1$
- $EV_{away} = (1-p) \cdot o_{away} - 1$

Policy:
- If $\max(EV_{home}, EV_{away}) \le \texttt{ev_threshold}$ → **no bet**
- Else bet the side with larger EV.

Profit for a $1 bet:
- Win → $o - 1$
- Lose → $-1$

ROI is computed as:

$$ROI = \frac{\text{total profit}}{\text{# bets (total stake)}}$$

---

## 6) How much data should you train on?

There isn’t one “correct” number; MLB betting is **non-stationary** (line-making, injuries, rule changes, teams change year to year). Use these practical rules:

### Recommended default
- Start with **3 seasons for training** (e.g., 2022–2024) and **1 season for testing** (e.g., 2025).

This is usually enough to:
- learn stable market patterns
- avoid being overly dominated by one season’s quirks

### When you should use MORE data
- Your model is high-variance (results swing wildly month-to-month)
- Learning curves show improvements as training size grows

### When you should use LESS / more recent data
- The model performs worse when you include older seasons
- You suspect regime shifts (pricing changes, rule changes)

### A good way to decide empirically (recommended)
Do a **walk-forward** evaluation:
- Train on 2022 → test on 2023
- Train on 2022–2023 → test on 2024
- Train on 2022–2024 → test on 2025

If ROI improves as you add seasons, more data helps. If ROI degrades, older seasons are hurting.

---

## 7) Training vs Testing data: what’s the advantage?

### Training data
- Used to fit model parameters.
- Also where you can do **time-series CV** and feature/model selection.

### Testing data (holdout)
- Used **only once** at the end to estimate real-world performance.
- Protects you from “overfitting to history” via repeated experiments.

### Why time-based splitting matters here
Odds and outcomes are time-ordered. Random shuffles can leak future patterns into training.

Recommended split:
- Choose a **date boundary** (e.g., `--test-start-date 2025-01-01`).
- Train on everything before.
- Test on everything after.

### Practical workflow (strongly recommended)
- **Train**: 2022–2023
- **Validation** (for choosing `ev_threshold`, model type, features): 2024
- **Test** (final report): 2025

This gives you a clean final result while still letting you tune decisions responsibly.

---

## 8) Tips for improving profit (next steps)

If you want higher profit/ROI, the biggest levers are usually:

- Tune `--ev-threshold` (higher threshold = fewer bets, potentially higher ROI)
- Compare `lr` vs `rf` vs `xgb`
- Use walk-forward evaluation rather than a single split
- Audit missing labels (non-final games / pushes) and ensure they’re excluded from training

---

## Troubleshooting

- XGBoost warning about glibc: it’s a system warning; LR/RF still work fine.

---

## 9) One-command experiment runner (for full report)

Script: `ml/run_experiments.py`

This runner automates report-ready experiments and writes CSV tables + plots covering:

- multi-model supervised learning (LR/RF and optional XGB)
- cross-validation metrics (time-series CV)
- test metrics: accuracy, precision, recall, F1, AUROC, logloss
- confusion matrix values (TN/FP/FN/TP)
- EV/ROI backtest metrics
- training data amount sensitivity
- class-balance methods (`class_weight`, `SMOTE`)
- simple tabular augmentation (numeric jitter)
- dimensionality reduction (PCA)

Run full experiment suite:

```bash
.venv/bin/python ml/run_experiments.py \
  --data all_2021_2025.csv \
  --out-dir reports/experiments \
  --test-year 2025 \
  --cv-folds 5 \
  --ev-threshold 0.0 \
  --verbose
```

Fast debug run:

```bash
.venv/bin/python ml/run_experiments.py --quick
```

Output artifacts (in `--out-dir`):

- `all_results.csv`
- `summary_by_experiment.csv`
- `confusion_and_metrics.csv`
- `plot_training_size_auc.png`, `plot_training_size_roi.png`
- `plot_balance_f1.png`, `plot_balance_roi.png`
- `plot_augmentation_auc.png`, `plot_augmentation_roi.png`
- `plot_pca_auc.png`, `plot_pca_roi.png`
