#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier  # type: ignore

    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


DEFAULT_DATA_CSV = Path(__file__).resolve().parents[2] / "dataset" / "all_2021_2025.csv"


def _parse_date_iso(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def _implied_prob(dec_odds: Any) -> Optional[float]:
    o = _safe_float(dec_odds)
    if o is None or o <= 1.0:
        return None
    return 1.0 / o


def _normalize_pair(p1: Optional[float], p2: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    if p1 is None or p2 is None:
        return None, None
    s = p1 + p2
    if s <= 0:
        return None, None
    return p1 / s, p2 / s


@dataclass(frozen=True)
class ProfitSummary:
    bets: int
    staked: float
    profit: float

    @property
    def roi(self) -> float:
        return 0.0 if self.staked <= 0 else self.profit / self.staked


def _profit_moneyline(
    *,
    p_home: float,
    odds_home: float,
    odds_away: float,
    home_win: int,
    ev_threshold: float,
) -> Optional[float]:
    """Return profit for a $1 bet under an EV-based policy; None means no bet."""

    p_home = float(np.clip(p_home, 1e-6, 1 - 1e-6))
    p_away = 1.0 - p_home

    ev_home = p_home * odds_home - 1.0
    ev_away = p_away * odds_away - 1.0

    if max(ev_home, ev_away) <= ev_threshold:
        return None

    bet_home = ev_home >= ev_away

    if bet_home:
        return (odds_home - 1.0) if home_win == 1 else -1.0
    return (odds_away - 1.0) if home_win == 0 else -1.0


def _profit_spread(
    *,
    p_home_cover: float,
    odds_home: float,
    odds_away: float,
    home_cover: int,
    ev_threshold: float,
) -> Optional[float]:
    """Return profit for a $1 spread bet under an EV-based policy; None means no bet."""

    p_home_cover = float(np.clip(p_home_cover, 1e-6, 1 - 1e-6))
    p_away_cover = 1.0 - p_home_cover

    ev_home = p_home_cover * odds_home - 1.0
    ev_away = p_away_cover * odds_away - 1.0

    if max(ev_home, ev_away) <= ev_threshold:
        return None

    bet_home = ev_home >= ev_away

    if bet_home:
        return (odds_home - 1.0) if home_cover == 1 else -1.0
    return (odds_away - 1.0) if home_cover == 0 else -1.0


def _summarize_profits(profits: list[Optional[float]]) -> ProfitSummary:
    realized = [p for p in profits if p is not None]
    staked = float(len(realized))
    profit = float(np.sum(realized)) if realized else 0.0
    return ProfitSummary(bets=len(realized), staked=staked, profit=profit)


def _build_model(model_name: str, *, random_state: int) -> Any:
    model_name = model_name.lower().strip()
    if model_name == "lr":
        return LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            n_jobs=None,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "xgb":
        if not _HAS_XGB:
            raise SystemExit("xgboost is not available in this environment")
        return XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
    raise SystemExit("Unknown --model. Use: lr | rf | xgb")


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["date_dt", "game_id"], kind="mergesort").reset_index(drop=True)

    # Fallbacks (enrichment might not populate everything if SBR JSON shape changes)
    if "ml_home_close_decimal" in out.columns:
        out["ml_home_close_use"] = out["ml_home_close_decimal"].fillna(out.get("moneyline_home_decimal"))
        out["ml_away_close_use"] = out["ml_away_close_decimal"].fillna(out.get("moneyline_away_decimal"))
    else:
        out["ml_home_close_use"] = out.get("moneyline_home_decimal")
        out["ml_away_close_use"] = out.get("moneyline_away_decimal")

    if "ps_home_close_decimal" in out.columns:
        out["ps_home_close_use"] = out["ps_home_close_decimal"].fillna(out.get("spread_home_decimal"))
        out["ps_away_close_use"] = out["ps_away_close_decimal"].fillna(out.get("spread_away_decimal"))
    else:
        out["ps_home_close_use"] = out.get("spread_home_decimal")
        out["ps_away_close_use"] = out.get("spread_away_decimal")

    # Market implied probs (vig-adjusted)
    out["ml_home_imp"] = out["ml_home_close_use"].map(_implied_prob)
    out["ml_away_imp"] = out["ml_away_close_use"].map(_implied_prob)

    nv = out.apply(lambda r: pd.Series(_normalize_pair(r["ml_home_imp"], r["ml_away_imp"])), axis=1)
    out["ml_home_imp_nv"], out["ml_away_imp_nv"] = nv.iloc[:, 0], nv.iloc[:, 1]

    # Date parts
    out["dow"] = out["date_dt"].dt.dayofweek
    out["month"] = out["date_dt"].dt.month
    out["year"] = out["date_dt"].dt.year

    # Labels: support direct crawl output (result/spread_result/scores) and older enriched data.
    if "home_win" in out.columns:
        out["home_win"] = pd.to_numeric(out["home_win"], errors="coerce")
    else:
        out["home_win"] = np.nan

    if "result" in out.columns:
        out.loc[out["result"].astype(str) == "home_win", "home_win"] = 1
        out.loc[out["result"].astype(str) == "away_win", "home_win"] = 0

    if "home_score" in out.columns and "away_score" in out.columns:
        hs = pd.to_numeric(out["home_score"], errors="coerce")
        aw = pd.to_numeric(out["away_score"], errors="coerce")
        mask_scores = hs.notna() & aw.notna()
        out.loc[mask_scores, "home_win"] = (hs[mask_scores] > aw[mask_scores]).astype(int)

    if "home_cover" in out.columns:
        out["home_cover"] = pd.to_numeric(out["home_cover"], errors="coerce")
    else:
        out["home_cover"] = np.nan

    if "spread_result" in out.columns:
        out.loc[out["spread_result"].astype(str) == "home_cover", "home_cover"] = 1
        out.loc[out["spread_result"].astype(str) == "away_cover", "home_cover"] = 0

    # Fallback spread label derivation from scores + home spread if spread_result is absent.
    if "home_score" in out.columns and "away_score" in out.columns and "home_spread" in out.columns:
        hs = pd.to_numeric(out["home_score"], errors="coerce")
        aw = pd.to_numeric(out["away_score"], errors="coerce")
        sp = pd.to_numeric(out["home_spread"], errors="coerce")
        mask_spread = hs.notna() & aw.notna() & sp.notna()
        margin = hs[mask_spread] + sp[mask_spread] - aw[mask_spread]
        out.loc[mask_spread & (margin > 0), "home_cover"] = 1
        out.loc[mask_spread & (margin < 0), "home_cover"] = 0

    return out


def train_eval(
    *,
    data_csv: Path,
    market: str,
    model_name: str,
    test_start_date: str,
    ev_threshold: float,
    cv_folds: int,
    random_state: int,
    out_predictions: Optional[Path],
) -> None:
    # Load and normalize schema so both crawl output and legacy enriched files are supported.
    df = pd.read_csv(data_csv)
    df = _prepare_frame(df)

    market = market.lower().strip()
    if market == "moneyline":
        label_col = "home_win"
        odds_home_col = "ml_home_close_use"
        odds_away_col = "ml_away_close_use"
    elif market == "spread":
        label_col = "home_cover"
        odds_home_col = "ps_home_close_use"
        odds_away_col = "ps_away_close_use"
    else:
        raise SystemExit("Unknown --market. Use: moneyline | spread")

    if label_col not in df.columns:
        raise SystemExit(f"Missing label column {label_col!r} in input data.")

    df = df[df["date_dt"].notna()].copy()
    df = df[df[label_col].isin([0, 1])].copy()

    df = df[df[odds_home_col].notna() & df[odds_away_col].notna()].copy()

    # Time-based holdout split: train before boundary, test on/after boundary.
    test_start = _parse_date_iso(test_start_date)
    is_test = df["date_dt"] >= test_start

    train_df = df[~is_test].copy()
    test_df = df[is_test].copy()

    if len(train_df) < 200 or len(test_df) < 50:
        raise SystemExit(f"Not enough rows after split: train={len(train_df)} test={len(test_df)}")

    # Baseline feature set from final_report.md: team IDs + odds/spread + date parts.
    categorical = ["away_team", "home_team"]

    if market == "moneyline":
        numeric = [
            odds_home_col,
            odds_away_col,
            "ml_home_imp_nv",
            "dow",
            "month",
            "year",
        ]
    else:
        numeric = [
            odds_home_col,
            odds_away_col,
            "home_spread",
            "away_spread",
            "dow",
            "month",
            "year",
        ]
    numeric = [c for c in numeric if c in df.columns]

    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = _build_model(model_name, random_state=random_state)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    X_train = train_df[categorical + numeric]
    y_train = train_df[label_col].astype(int)

    X_test = test_df[categorical + numeric]
    y_test = test_df[label_col].astype(int)

    # Optional time-series CV reports pre-holdout calibration quality.
    if cv_folds > 1:
        splitter = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        y_train_arr = y_train.to_numpy()
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_train), start=1):
            pipe.fit(X_train.iloc[tr_idx], y_train_arr[tr_idx])
            p_va = pipe.predict_proba(X_train.iloc[va_idx])[:, 1]
            ll = log_loss(y_train_arr[va_idx], p_va)
            cv_scores.append(ll)
        print(f"CV logloss (train, {cv_folds} folds): mean={np.mean(cv_scores):.4f} std={np.std(cv_scores):.4f}")

    pipe.fit(X_train, y_train)

    p_test = pipe.predict_proba(X_test)[:, 1]
    pred_test = (p_test >= 0.5).astype(int)

    ll = log_loss(y_test, p_test)
    acc = accuracy_score(y_test, pred_test)
    try:
        auc = roc_auc_score(y_test, p_test)
    except Exception:
        auc = float("nan")

    print(f"Test rows: {len(test_df)} (from {test_start_date})")
    print(f"Model: {model_name}  Market: {market}")
    print(f"LogLoss: {ll:.4f}  AUC: {auc:.4f}  Acc: {acc:.4f}")

    odds_home = test_df[odds_home_col].astype(float).to_numpy()
    odds_away = test_df[odds_away_col].astype(float).to_numpy()
    y_true = y_test.to_numpy()

    # Holdout EV backtest for selected market.
    if market == "moneyline":
        profits = [
            _profit_moneyline(
                p_home=float(p_test[i]),
                odds_home=float(odds_home[i]),
                odds_away=float(odds_away[i]),
                home_win=int(y_true[i]),
                ev_threshold=float(ev_threshold),
            )
            for i in range(len(test_df))
        ]

        # Baseline benchmark: vig-adjusted implied probability from market odds.
        base_p = test_df["ml_home_imp_nv"].astype(float).to_numpy()
        base_profits = [
            _profit_moneyline(
                p_home=float(base_p[i]),
                odds_home=float(odds_home[i]),
                odds_away=float(odds_away[i]),
                home_win=int(y_true[i]),
                ev_threshold=float(ev_threshold),
            )
            for i in range(len(test_df))
        ]

    else:
        profits = [
            _profit_spread(
                p_home_cover=float(p_test[i]),
                odds_home=float(odds_home[i]),
                odds_away=float(odds_away[i]),
                home_cover=int(y_true[i]),
                ev_threshold=float(ev_threshold),
            )
            for i in range(len(test_df))
        ]
        # Baseline benchmark for spread: 0.5 probability (no informational edge).
        base_profits = [
            _profit_spread(
                p_home_cover=0.5,
                odds_home=float(odds_home[i]),
                odds_away=float(odds_away[i]),
                home_cover=int(y_true[i]),
                ev_threshold=float(ev_threshold),
            )
            for i in range(len(test_df))
        ]

    s = _summarize_profits(profits)
    b = _summarize_profits(base_profits)

    print(f"EV threshold: {ev_threshold:.4f} (bet if max(EV) > threshold)")
    print(f"Strategy (model): bets={s.bets} profit={s.profit:.2f} ROI={s.roi*100:.2f}%")
    print(f"Baseline:          bets={b.bets} profit={b.profit:.2f} ROI={b.roi*100:.2f}%")

    # Optional per-game audit file (probability, EV, pick side, and realized profit).
    if out_predictions is not None:
        out = test_df[["date", "game_id", "away_team", "home_team"]].copy()
        p_col = "p_home" if market == "moneyline" else "p_home_cover"
        out[p_col] = p_test
        out["y_true"] = y_true
        out["odds_home"] = odds_home
        out["odds_away"] = odds_away

        p_home = p_test
        p_away = 1.0 - p_home
        ev_home = p_home * odds_home - 1.0
        ev_away = p_away * odds_away - 1.0
        out["ev_home"] = ev_home
        out["ev_away"] = ev_away
        out["pick_side"] = np.where(ev_home >= ev_away, "home", "away")
        out["bet"] = np.where(np.maximum(ev_home, ev_away) > float(ev_threshold), 1, 0)

        out["profit"] = [p if p is not None else 0.0 for p in profits]
        out_predictions.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_predictions, index=False)
        print(f"Wrote predictions -> {out_predictions}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train a classifier and evaluate using profit/ROI (not just accuracy)."
    )
    ap.add_argument(
        "--data",
        default=str(DEFAULT_DATA_CSV),
        help="Input CSV (supports direct crawl output with result/spread_result or legacy enriched labels)",
    )
    ap.add_argument("--market", default="moneyline", choices=["moneyline", "spread"])
    ap.add_argument("--model", default="lr", choices=["lr", "rf", "xgb"])
    ap.add_argument(
        "--test-start-date",
        default="2025-01-01",
        help="YYYY-MM-DD; rows on/after go to test set",
    )
    ap.add_argument(
        "--ev-threshold",
        type=float,
        default=0.0,
        help="Only bet when expected value exceeds this (stake=1).",
    )
    ap.add_argument("--cv-folds", type=int, default=0, help="TimeSeries CV folds on train set")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--out-predictions", default="", help="Optional CSV to write test predictions")
    args = ap.parse_args()

    out_pred = Path(args.out_predictions) if args.out_predictions.strip() else None

    train_eval(
        data_csv=Path(args.data),
        market=str(args.market),
        model_name=str(args.model),
        test_start_date=str(args.test_start_date),
        ev_threshold=float(args.ev_threshold),
        cv_folds=int(args.cv_folds),
        random_state=int(args.random_state),
        out_predictions=out_pred,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
