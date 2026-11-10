"""Train and evaluate engagement models on brain features vs. baselines.

The point of this module is the *comparison*, not any single model. A
brain-derived feature only earns its place if it improves out-of-sample
prediction over what you already have (duration, creator, metadata, the
raw video/audio/text embeddings TRIBE itself is built on). Every evaluation
therefore reports each feature set side by side on identical folds.

Folds are grouped (by creator for video-level models, by video for
per-second models) so the model is always scored on videos or creators it
never saw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str = "ridge", seed: int = 0) -> RegressorMixin:
    if kind == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(f"Unknown model kind {kind!r} (use 'ridge' or 'hgb')")


def creator_normalize(
    df: pd.DataFrame, target: str, creator_col: str = "creator_id", log: bool = False
) -> pd.Series:
    """Z-score ``target`` within each creator.

    Raw performance is dominated by audience size: a mediocre video from a big
    account beats a great one from a small account. Normalizing within creator
    asks the question content decisions can actually affect: "is this video
    better or worse than this creator's usual?". Creators with a single video
    get 0 (no within-creator information).
    """
    y = np.log1p(df[target].astype(float)) if log else df[target].astype(float)
    grouped = y.groupby(df[creator_col])
    sd = grouped.transform(lambda s: s.std(ddof=0))
    z = (y - grouped.transform("mean")) / sd.replace(0, np.nan)
    return z.fillna(0.0).rename(f"{target}__creator_z")


@dataclass
class EvalResult:
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    summary: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.summary = summarize(self.fold_metrics)


def summarize(fold_metrics: pd.DataFrame, baseline: str | None = None) -> pd.DataFrame:
    """Mean/std of each metric per feature set, plus gain over the first set."""
    metrics = [c for c in fold_metrics.columns if c not in {"feature_set", "fold"}]
    agg = fold_metrics.groupby("feature_set", sort=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    baseline = baseline or fold_metrics["feature_set"].iloc[0]
    primary = metrics[0]
    base = fold_metrics[fold_metrics.feature_set == baseline].set_index("fold")[primary]
    gains, wins = [], []
    for name in agg.index:
        cur = fold_metrics[fold_metrics.feature_set == name].set_index("fold")[primary]
        diff = (cur - base).dropna()
        gains.append(diff.mean())
        wins.append((diff > 0).mean())
    agg[f"{primary}_gain_vs_{baseline}"] = gains
    agg[f"folds_better_than_{baseline}"] = wins
    return agg.reset_index()


def _check_sets(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> None:
    for name, cols in feature_sets.items():
        if not cols:
            raise ValueError(f"Feature set {name!r} is empty")
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Feature set {name!r} missing columns: {missing[:5]}")


def evaluate_video_model(
    data: pd.DataFrame,
    target: str,
    feature_sets: dict[str, list[str]],
    group_col: str | None = "creator_id",
    n_splits: int = 5,
    model: str = "ridge",
    seed: int = 0,
) -> EvalResult:
    """Cross-validation of each feature set for one video-level target.

    With ``group_col`` (e.g. creator), folds hold out whole groups. With
    ``group_col=None`` (datasets without creator IDs, like SnapUGC), folds
    are shuffled K-fold over videos.

    Metrics per fold: Spearman rank correlation (does the model *order*
    videos correctly? usually what matters for "which draft should I post")
    and R^2.
    """
    _check_sets(data, feature_sets)
    data = data[data[target].notna()].reset_index(drop=True)
    y = data[target].to_numpy(float)
    if group_col is None:
        n_splits = min(n_splits, len(data))
        folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(data))
    else:
        groups = data[group_col].to_numpy()
        n_splits = min(n_splits, len(np.unique(groups)))
        folds = list(GroupKFold(n_splits=n_splits).split(data, y, groups))
    rows, oof = [], {"video_id": data["video_id"], "y_true": y}
    for name, cols in feature_sets.items():
        X = data[cols].to_numpy(float)
        pred = np.full(len(y), np.nan)
        for k, (tr, te) in enumerate(folds):
            est = clone(make_model(model, seed)).fit(X[tr], y[tr])
            pred[te] = est.predict(X[te])
            rows.append(
                {
                    "feature_set": name,
                    "fold": k,
                    "spearman": spearmanr(y[te], pred[te]).statistic,
                    "r2": r2_score(y[te], pred[te]),
                }
            )
        oof[f"pred__{name}"] = pred
    return EvalResult(pd.DataFrame(rows), pd.DataFrame(oof))


def evaluate_dropoff_model(
    data: pd.DataFrame,
    target: str,
    feature_sets: dict[str, list[str]],
    group_col: str = "video_id",
    n_splits: int = 5,
    model: str = "hgb",
    seed: int = 0,
    skip_first_seconds: int = 3,
) -> EvalResult:
    """Grouped CV for per-second drop-off (hazard) prediction.

    Metrics per fold:

    * ``within_video_spearman``: mean over held-out videos of the rank
      correlation between predicted and true hazard across that video's
      seconds, ignoring the first ``skip_first_seconds`` (every video loses
      most viewers at the start, which would make any model look good).
      This answers "does it find the moments people leave?".
    * ``r2``: pooled R^2 over all held-out seconds.
    """
    _check_sets(data, feature_sets)
    data = data.reset_index(drop=True)
    y = data[target].to_numpy(float)
    groups = data[group_col].to_numpy()
    n_splits = min(n_splits, len(np.unique(groups)))
    folds = list(GroupKFold(n_splits=n_splits).split(data, y, groups))
    late = (data["t"] - data.groupby("video_id")["t"].transform("min")) >= skip_first_seconds
    rows, oof = [], {"video_id": data["video_id"], "t": data["t"], "y_true": y}
    for name, cols in feature_sets.items():
        X = data[cols].to_numpy(float)
        pred = np.full(len(y), np.nan)
        for k, (tr, te) in enumerate(folds):
            est = clone(make_model(model, seed)).fit(X[tr], y[tr])
            pred[te] = est.predict(X[te])
            test = data.iloc[te].assign(pred=pred[te], y=y[te])[late.iloc[te].to_numpy()]
            per_video = [
                spearmanr(g["y"], g["pred"]).statistic
                for _, g in test.groupby("video_id")
                if len(g) > 3 and g["y"].nunique() > 1 and g["pred"].nunique() > 1
            ]
            rows.append(
                {
                    "feature_set": name,
                    "fold": k,
                    "within_video_spearman": float(np.nanmean(per_video)) if per_video else np.nan,
                    "r2": r2_score(y[te], pred[te]),
                }
            )
        oof[f"pred__{name}"] = pred
    return EvalResult(pd.DataFrame(rows), pd.DataFrame(oof))


def fit_final(data: pd.DataFrame, target: str, cols: list[str], model: str, seed: int = 0):
    """Fit on all rows, for scoring new videos."""
    est = make_model(model, seed)
    return est.fit(data[cols].to_numpy(float), data[target].to_numpy(float))
