#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier  # type: ignore

    HAS_XGB = True
except Exception:
    HAS_XGB = False


@dataclass(frozen=True)
class EvalResult:
    experiment: str
    setting: str
    model: str
    train_years: str
    train_rows: int
    test_rows: int
    cv_folds: int
    cv_acc_mean: float
    cv_prec_mean: float
    cv_rec_mean: float
    cv_f1_mean: float
    cv_auc_mean: float
    cv_logloss_mean: float
    test_acc: float
    test_prec: float
    test_rec: float
    test_f1: float
    test_auc: float
    test_logloss: float
    tn: int
    fp: int
    fn: int
    tp: int
    bets: int
    profit: float
    roi: float


def _log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[INFO {now}] {msg}")


def _build_model(name: str, *, random_state: int, class_weight_mode: str) -> Any:
    class_weight: Optional[str] = None
    if class_weight_mode == "balanced":
        class_weight = "balanced"

    if name == "lr":
        return LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight=class_weight,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
            class_weight=class_weight,
        )
    if name == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed")
        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
    raise ValueError(f"Unsupported model {name!r}")


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")
    out = out[out["date_dt"].notna()].copy()
    out = out.sort_values(["date_dt", "game_id"], kind="mergesort").reset_index(drop=True)

    out["dow"] = out["date_dt"].dt.dayofweek
    out["month"] = out["date_dt"].dt.month
    out["year"] = out["date_dt"].dt.year

    # fallback odds columns
    out["ml_home_close_use"] = out.get("ml_home_close_decimal", out.get("moneyline_home_decimal"))
    out["ml_away_close_use"] = out.get("ml_away_close_decimal", out.get("moneyline_away_decimal"))
    out["spread_home_use"] = out.get("spread_home_decimal")
    out["spread_away_use"] = out.get("spread_away_decimal")

    out["ml_home_close_use"] = out["ml_home_close_use"].fillna(out.get("moneyline_home_decimal"))
    out["ml_away_close_use"] = out["ml_away_close_use"].fillna(out.get("moneyline_away_decimal"))

    if "home_win" in out.columns:
        out["home_win"] = pd.to_numeric(out["home_win"], errors="coerce")
    else:
        out["home_win"] = np.nan

    # Backward-compatible label derivation from crawl output.
    if "result" in out.columns:
        out.loc[out["result"].astype(str) == "home_win", "home_win"] = 1
        out.loc[out["result"].astype(str) == "away_win", "home_win"] = 0

    if "away_score" in out.columns and "home_score" in out.columns:
        hs = pd.to_numeric(out["home_score"], errors="coerce")
        aw = pd.to_numeric(out["away_score"], errors="coerce")
        valid_scores = hs.notna() & aw.notna()
        out.loc[valid_scores, "home_win"] = (hs[valid_scores] > aw[valid_scores]).astype(int)

    # Spread label for spread-target experiments.
    if "home_cover" in out.columns:
        out["home_cover"] = pd.to_numeric(out["home_cover"], errors="coerce")
    else:
        out["home_cover"] = np.nan

    if "spread_result" in out.columns:
        out.loc[out["spread_result"].astype(str) == "home_cover", "home_cover"] = 1
        out.loc[out["spread_result"].astype(str) == "away_cover", "home_cover"] = 0

    if "away_score" in out.columns and "home_score" in out.columns and "home_spread" in out.columns:
        hs = pd.to_numeric(out["home_score"], errors="coerce")
        aw = pd.to_numeric(out["away_score"], errors="coerce")
        hsp = pd.to_numeric(out["home_spread"], errors="coerce")
        valid_spread = hs.notna() & aw.notna() & hsp.notna()
        margin = hs[valid_spread] + hsp[valid_spread] - aw[valid_spread]
        out.loc[valid_spread & (margin > 0), "home_cover"] = 1
        out.loc[valid_spread & (margin < 0), "home_cover"] = 0

    return out


def _make_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                cat_pipe,
                categorical,
            ),
            (
                "num",
                num_pipe,
                numeric,
            ),
        ],
        remainder="drop",
    )


def _augment_tabular(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    numeric_for_noise: list[str],
    noise_std: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) == 0:
        return X, y

    rng = np.random.default_rng(random_state)
    X_aug = X.copy()

    for col in numeric_for_noise:
        if col not in X_aug.columns:
            continue
        vals = pd.to_numeric(X_aug[col], errors="coerce")
        scale = float(np.nanstd(vals.to_numpy()))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        noise = rng.normal(loc=0.0, scale=noise_std * scale, size=len(X_aug))
        X_aug[col] = vals.fillna(0.0) + noise

    X_out = pd.concat([X, X_aug], ignore_index=True)
    y_out = pd.concat([y, y], ignore_index=True)
    return X_out, y_out


def _profit_summary(
    probs_home: np.ndarray,
    odds_home: np.ndarray,
    odds_away: np.ndarray,
    y_true: np.ndarray,
    *,
    ev_threshold: float,
) -> tuple[int, float, float]:
    p_home = np.clip(probs_home.astype(float), 1e-6, 1.0 - 1e-6)
    p_away = 1.0 - p_home

    ev_home = p_home * odds_home - 1.0
    ev_away = p_away * odds_away - 1.0

    best_is_home = ev_home >= ev_away
    best_ev = np.where(best_is_home, ev_home, ev_away)
    bet_mask = best_ev > float(ev_threshold)

    profits = np.zeros_like(p_home, dtype=float)

    win_home = y_true.astype(int) == 1
    win_away = ~win_home

    home_profit = np.where(win_home, odds_home - 1.0, -1.0)
    away_profit = np.where(win_away, odds_away - 1.0, -1.0)
    pick_profit = np.where(best_is_home, home_profit, away_profit)

    profits[bet_mask] = pick_profit[bet_mask]

    bets = int(np.sum(bet_mask))
    total_profit = float(np.sum(profits))
    roi = 0.0 if bets == 0 else total_profit / float(bets)
    return bets, total_profit, roi


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _fit_eval(
    *,
    df: pd.DataFrame,
    train_end_year: int,
    test_year: int,
    model_name: str,
    experiment: str,
    setting: str,
    class_weight_mode: str,
    use_smote: bool,
    use_aug: bool,
    use_pca: bool,
    ev_threshold: float,
    cv_folds: int,
    random_state: int,
    verbose: bool,
    label_col: str = "home_win",
    odds_home_col: str = "ml_home_close_use",
    odds_away_col: str = "ml_away_close_use",
) -> EvalResult:
    if verbose:
        _log(
            f"start fit_eval experiment={experiment} setting={setting} model={model_name} "
            f"train<= {train_end_year} test={test_year}"
        )

    work = df.copy()
    work = work[work[label_col].isin([0, 1])].copy()

    train_mask = work["year"] <= train_end_year
    test_mask = work["year"] == test_year

    train_df = work[train_mask].copy()
    test_df = work[test_mask].copy()

    if len(train_df) < 200 or len(test_df) < 50:
        raise RuntimeError(
            f"Not enough data for split train<= {train_end_year}, test={test_year}: "
            f"train={len(train_df)} test={len(test_df)}"
        )

    categorical = ["away_team", "home_team"]
    numeric = [
        "ml_home_close_use",
        "ml_away_close_use",
        "home_spread",
        "away_spread",
        "spread_home_decimal",
        "spread_away_decimal",
        "dow",
        "month",
        "year",
    ]
    numeric = [c for c in numeric if c in work.columns]

    feature_cols = categorical + numeric

    X_train = train_df[feature_cols].copy()
    y_train = train_df[label_col].astype(int).copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[label_col].astype(int).to_numpy()

    pre = _make_preprocessor(categorical, numeric)
    clf = _build_model(model_name, random_state=random_state, class_weight_mode=class_weight_mode)

    steps: list[tuple[str, Any]] = [("pre", pre)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, random_state=random_state)))
    steps.append(("clf", clf))

    pipe = ImbPipeline(steps=steps)

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    cv_metrics: list[dict[str, float]] = []

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    for tr_idx, va_idx in tscv.split(X_train):
        X_tr = X_train.iloc[tr_idx].copy()
        y_tr = y_train.iloc[tr_idx].copy()
        X_va = X_train.iloc[va_idx].copy()
        y_va = y_train.iloc[va_idx].to_numpy()

        if use_aug:
            X_tr, y_tr = _augment_tabular(
                X_tr,
                y_tr,
                numeric_for_noise=[
                    "ml_home_close_use",
                    "ml_away_close_use",
                    "home_spread",
                    "away_spread",
                    "spread_home_decimal",
                    "spread_away_decimal",
                ],
                noise_std=0.01,
                random_state=random_state,
            )

        pipe.fit(X_tr, y_tr)
        p_va = pipe.predict_proba(X_va)[:, 1]
        y_hat = (p_va >= 0.5).astype(int)

        cv_metrics.append(
            {
                "acc": float(accuracy_score(y_va, y_hat)),
                "prec": float(precision_score(y_va, y_hat, zero_division=0)),
                "rec": float(recall_score(y_va, y_hat, zero_division=0)),
                "f1": float(f1_score(y_va, y_hat, zero_division=0)),
                "auc": _safe_auc(y_va, p_va),
                "logloss": float(log_loss(y_va, p_va, labels=[0, 1])),
            }
        )

    if use_aug:
        X_train_fit, y_train_fit = _augment_tabular(
            X_train,
            y_train,
            numeric_for_noise=[
                "ml_home_close_use",
                "ml_away_close_use",
                "home_spread",
                "away_spread",
                "spread_home_decimal",
                "spread_away_decimal",
            ],
            noise_std=0.01,
            random_state=random_state,
        )
    else:
        X_train_fit, y_train_fit = X_train, y_train

    pipe.fit(X_train_fit, y_train_fit)

    p_test = pipe.predict_proba(X_test)[:, 1]
    y_pred = (p_test >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = _safe_auc(y_test, p_test)
    ll = float(log_loss(y_test, p_test, labels=[0, 1]))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel().tolist()

    bets, profit, roi = _profit_summary(
        probs_home=p_test,
        odds_home=test_df[odds_home_col].astype(float).to_numpy(),
        odds_away=test_df[odds_away_col].astype(float).to_numpy(),
        y_true=y_test,
        ev_threshold=ev_threshold,
    )

    train_years = f"<= {train_end_year}"

    if verbose:
        _log(
            f"done fit_eval experiment={experiment} setting={setting} model={model_name} "
            f"test_auc={auc:.4f} test_f1={f1:.4f} roi={roi:.4f} bets={bets}"
        )

    return EvalResult(
        experiment=experiment,
        setting=setting,
        model=model_name,
        train_years=train_years,
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        cv_folds=int(cv_folds),
        cv_acc_mean=float(np.mean([m["acc"] for m in cv_metrics])),
        cv_prec_mean=float(np.mean([m["prec"] for m in cv_metrics])),
        cv_rec_mean=float(np.mean([m["rec"] for m in cv_metrics])),
        cv_f1_mean=float(np.mean([m["f1"] for m in cv_metrics])),
        cv_auc_mean=float(np.mean([m["auc"] for m in cv_metrics])),
        cv_logloss_mean=float(np.mean([m["logloss"] for m in cv_metrics])),
        test_acc=acc,
        test_prec=prec,
        test_rec=rec,
        test_f1=f1,
        test_auc=auc,
        test_logloss=ll,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        bets=int(bets),
        profit=float(profit),
        roi=float(roi),
    )


def _save_results(results: list[EvalResult], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(out_dir / "all_results.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return df


def _plot_training_size(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["experiment"] == "training_size"].copy()
    if sub.empty:
        return

    sub["train_rows_k"] = sub["train_rows"] / 1000.0

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=sub, x="train_rows_k", y="test_auc", hue="model", marker="o")
    plt.title("AUC vs Training Size")
    plt.xlabel("Training rows (thousands)")
    plt.ylabel("Test AUROC")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_training_size_auc.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=sub, x="train_rows_k", y="roi", hue="model", marker="o")
    plt.title("ROI vs Training Size")
    plt.xlabel("Training rows (thousands)")
    plt.ylabel("ROI")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_training_size_roi.png", dpi=150)
    plt.close()


def _plot_balance(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["experiment"] == "balance"].copy()
    if sub.empty:
        return

    plt.figure(figsize=(10, 5))
    sns.barplot(data=sub, x="setting", y="test_f1", hue="model")
    plt.title("Balance Strategies: Test F1")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_balance_f1.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=sub, x="setting", y="roi", hue="model")
    plt.title("Balance Strategies: ROI")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_balance_roi.png", dpi=150)
    plt.close()


def _plot_augmentation(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["experiment"] == "augmentation"].copy()
    if sub.empty:
        return

    plt.figure(figsize=(8, 5))
    sns.barplot(data=sub, x="setting", y="test_auc", hue="model")
    plt.title("Augmentation: Test AUROC")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_augmentation_auc.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=sub, x="setting", y="roi", hue="model")
    plt.title("Augmentation: ROI")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_augmentation_roi.png", dpi=150)
    plt.close()


def _plot_pca(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["experiment"] == "pca"].copy()
    if sub.empty:
        return

    plt.figure(figsize=(8, 5))
    sns.barplot(data=sub, x="setting", y="test_auc", hue="model")
    plt.title("Dimensionality Reduction: Test AUROC")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_pca_auc.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=sub, x="setting", y="roi", hue="model")
    plt.title("Dimensionality Reduction: ROI")
    plt.tight_layout()
    plt.savefig(out_dir / "plot_pca_roi.png", dpi=150)
    plt.close()


def _save_confusion_tables(df: pd.DataFrame, out_dir: Path) -> None:
    keep = [
        "experiment",
        "setting",
        "model",
        "tn",
        "fp",
        "fn",
        "tp",
        "test_acc",
        "test_prec",
        "test_rec",
        "test_f1",
        "test_auc",
        "test_logloss",
        "roi",
    ]
    out = df[keep].copy()
    out.to_csv(out_dir / "confusion_and_metrics.csv", index=False)


def _make_summary(df: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        df.groupby(["experiment", "setting", "model"], as_index=False)
        .agg(
            cv_auc_mean=("cv_auc_mean", "mean"),
            test_auc=("test_auc", "mean"),
            test_f1=("test_f1", "mean"),
            roi=("roi", "mean"),
            profit=("profit", "mean"),
            bets=("bets", "mean"),
        )
        .sort_values(["experiment", "test_auc"], ascending=[True, False])
    )
    summary.to_csv(out_dir / "summary_by_experiment.csv", index=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run report-ready supervised learning experiments (CV metrics, confusion matrices, "
            "training-size, class balance, augmentation, PCA, and EV/ROI backtests)."
        )
    )
    ap.add_argument(
        "--data",
        default="2021.csv",
        help="Input CSV path (must already include labels like home_win, or scores/result columns).",
    )
    ap.add_argument("--out-dir", default="reports/experiments", help="Output directory for tables and plots.")
    ap.add_argument("--test-year", type=int, default=2025, help="Holdout test year.")
    ap.add_argument("--cv-folds", type=int, default=5, help="TimeSeries CV folds.")
    ap.add_argument("--ev-threshold", type=float, default=0.0, help="EV threshold for bet/no-bet policy.")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Run a lighter subset of experiments for faster iteration.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-run progress lines while experiments are executing.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    started_at = datetime.now()
    _log("experiment runner started")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    if not data_path.exists():
        raise RuntimeError(f"Input data file not found: {data_path}")

    _log("loading input dataset")
    df = pd.read_csv(data_path)
    df = _prepare_frame(df)
    _log(f"dataset ready rows={len(df)} years={sorted(df['year'].dropna().unique().tolist())}")

    if df["year"].nunique() < 3:
        raise RuntimeError("Need at least 3 seasons for report experiments.")

    models: list[str] = ["lr", "rf"]
    if HAS_XGB and not args.quick:
        models.append("xgb")

    results: list[EvalResult] = []

    # 1) Baseline model comparison (also yields required multi-method CV metrics)
    _log(f"experiment: baseline model comparison ({len(models)} models)")
    for model in models:
        results.append(
            _fit_eval(
                df=df,
                train_end_year=args.test_year - 1,
                test_year=args.test_year,
                model_name=model,
                experiment="baseline",
                setting="default",
                class_weight_mode="none",
                use_smote=False,
                use_aug=False,
                use_pca=False,
                ev_threshold=float(args.ev_threshold),
                cv_folds=int(args.cv_folds),
                random_state=int(args.random_state),
                verbose=bool(args.verbose),
            )
        )

    # 2) Training-size experiment
    start_year = int(df["year"].min())
    end_train = args.test_year - 1
    train_ends = list(range(start_year, end_train + 1))
    if args.quick and len(train_ends) > 2:
        train_ends = [train_ends[0], train_ends[-1]]

    for train_end in train_ends:
        _log(f"experiment: training_size with train_end={train_end} ({len(models)} models)")
        for model in models:
            results.append(
                _fit_eval(
                    df=df,
                    train_end_year=train_end,
                    test_year=args.test_year,
                    model_name=model,
                    experiment="training_size",
                    setting=f"train<= {train_end}",
                    class_weight_mode="none",
                    use_smote=False,
                    use_aug=False,
                    use_pca=False,
                    ev_threshold=float(args.ev_threshold),
                    cv_folds=int(args.cv_folds),
                    random_state=int(args.random_state),
                    verbose=bool(args.verbose),
                )
            )

    # 3) Balance/composition experiment
    balance_models = ["lr", "rf"]
    if args.quick:
        balance_models = ["lr"]

    balance_settings = [
        ("none", "none", False),
        ("class_weight", "balanced", False),
        ("smote", "none", True),
    ]

    for model in balance_models:
        _log(f"experiment: balance for model={model} ({len(balance_settings)} settings)")
        for setting_name, cw_mode, use_smote in balance_settings:
            results.append(
                _fit_eval(
                    df=df,
                    train_end_year=args.test_year - 1,
                    test_year=args.test_year,
                    model_name=model,
                    experiment="balance",
                    setting=setting_name,
                    class_weight_mode=cw_mode,
                    use_smote=use_smote,
                    use_aug=False,
                    use_pca=False,
                    ev_threshold=float(args.ev_threshold),
                    cv_folds=int(args.cv_folds),
                    random_state=int(args.random_state),
                    verbose=bool(args.verbose),
                )
            )

    # 4) Data augmentation experiment (tabular jitter on train only)
    aug_models = ["lr", "rf"]
    if args.quick:
        aug_models = ["lr"]

    for model in aug_models:
        _log(f"experiment: augmentation for model={model}")
        for setting_name, use_aug in [("no_aug", False), ("noise_aug", True)]:
            results.append(
                _fit_eval(
                    df=df,
                    train_end_year=args.test_year - 1,
                    test_year=args.test_year,
                    model_name=model,
                    experiment="augmentation",
                    setting=setting_name,
                    class_weight_mode="none",
                    use_smote=False,
                    use_aug=use_aug,
                    use_pca=False,
                    ev_threshold=float(args.ev_threshold),
                    cv_folds=int(args.cv_folds),
                    random_state=int(args.random_state),
                    verbose=bool(args.verbose),
                )
            )

    # 5) Dimensionality reduction experiment
    pca_models = ["lr", "rf"]
    if args.quick:
        pca_models = ["lr"]

    for model in pca_models:
        _log(f"experiment: pca for model={model}")
        for setting_name, use_pca in [("no_pca", False), ("pca_95var", True)]:
            results.append(
                _fit_eval(
                    df=df,
                    train_end_year=args.test_year - 1,
                    test_year=args.test_year,
                    model_name=model,
                    experiment="pca",
                    setting=setting_name,
                    class_weight_mode="none",
                    use_smote=False,
                    use_aug=False,
                    use_pca=use_pca,
                    ev_threshold=float(args.ev_threshold),
                    cv_folds=int(args.cv_folds),
                    random_state=int(args.random_state),
                    verbose=bool(args.verbose),
                )
            )

    result_df = _save_results(results, out_dir)
    _save_confusion_tables(result_df, out_dir)
    _make_summary(result_df, out_dir)

    _plot_training_size(result_df, out_dir)
    _plot_balance(result_df, out_dir)
    _plot_augmentation(result_df, out_dir)
    _plot_pca(result_df, out_dir)

    elapsed = (datetime.now() - started_at).total_seconds()
    _log(f"all experiments complete in {elapsed:.1f}s")
    print(f"Wrote report artifacts to: {out_dir}")
    print("- all_results.csv")
    print("- summary_by_experiment.csv")
    print("- confusion_and_metrics.csv")
    print("- plot_training_size_auc.png, plot_training_size_roi.png")
    print("- plot_balance_f1.png, plot_balance_roi.png")
    print("- plot_augmentation_auc.png, plot_augmentation_roi.png")
    print("- plot_pca_auc.png, plot_pca_roi.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
