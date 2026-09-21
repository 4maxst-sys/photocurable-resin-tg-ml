"""Train the censored-Tg model and rank candidate resin formulations.

This is the standalone, command-line version of ``ML_formulations_3.ipynb``.
It has no Google Colab dependencies and reads all input data from local paths.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from math import erf, sqrt
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Config:
    sigma_k: float = 8.2
    censoring_threshold_k: float = 243.15
    em_iterations: int = 7
    random_seed: int = 42
    prediction_times_s: tuple[float, float, float] = (5.0, 10.0, 15.0)
    candidate_count: int = 200_000
    top_n: int = 5
    min_component_wt: float = 5.0
    max_component_wt: float = 80.0
    initiator_min_wt: float = 1.5
    initiator_max_wt: float = 4.0
    min_l1_distance: float = 18.0
    bounds_penalty_weight: float = 25.0
    closeness_penalty_weight: float = 15.0
    catboost_iterations: int = 2_000
    learning_rate: float = 0.03
    depth: int = 8
    l2_leaf_reg: float = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a censored CatBoost Tg model and rank new formulations."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("training_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--sigma-k", type=float, default=8.2)
    parser.add_argument("--censoring-threshold-k", type=float, default=243.15)
    parser.add_argument("--em-iterations", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--times",
        type=float,
        nargs=3,
        metavar=("T1", "T2", "T3"),
        default=(5.0, 10.0, 15.0),
        help="Three early prediction times in seconds; the fourth is dataset maximum.",
    )
    parser.add_argument("--candidates", type=int, default=200_000)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--catboost-iterations", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress periodic CatBoost training messages.",
    )
    return parser.parse_args()


def read_csv_sc(path: Path) -> pd.DataFrame:
    """Read the project's semicolon-separated, decimal-comma CSV format."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, sep=";", decimal=",", engine="python")


def find_col(
    frame: pd.DataFrame, patterns: Sequence[str], prefer: str | None = None
) -> str | None:
    if prefer and prefer in frame.columns:
        return prefer
    regex = re.compile("|".join(patterns), re.IGNORECASE)
    return next((column for column in frame.columns if regex.search(column)), None)


def norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def truncated_normal_mean_left(
    mu: np.ndarray, sigma: float, upper: float, clip_min: float = 120.0
) -> np.ndarray:
    alpha = (upper - mu) / sigma
    phi = np.exp(-0.5 * alpha**2) / np.sqrt(2.0 * np.pi)
    probability = norm_cdf(alpha)
    conditional_mean = mu - sigma * phi / np.clip(probability, 1e-12, None)
    return np.maximum(conditional_mean, clip_min)


def build_form_features(
    formulations: pd.DataFrame, properties: pd.DataFrame
) -> tuple[pd.DataFrame, str, list[str], list[str]]:
    sample_col = "Sample_ID"
    key_col = "Component"
    if sample_col not in formulations:
        raise ValueError(f"{sample_col!r} is missing from formulations file")
    if key_col not in properties:
        raise ValueError(f"{key_col!r} is missing from monomer properties file")

    weight_cols = [column for column in formulations.columns if column != sample_col]
    numeric_property_cols = [
        column
        for column in properties.columns
        if column != key_col and pd.api.types.is_numeric_dtype(properties[column])
    ]
    if not numeric_property_cols:
        raise ValueError("No numeric monomer properties were found")

    long = formulations[[sample_col, *weight_cols]].melt(
        sample_col, var_name=key_col, value_name="wt"
    )
    long["wt"] = pd.to_numeric(long["wt"], errors="coerce").fillna(0.0)
    merged = long.merge(
        properties[[key_col, *numeric_property_cols]], on=key_col, how="left"
    )
    missing = sorted(set(weight_cols) - set(properties[key_col].astype(str)))
    if missing:
        raise ValueError(
            "Components missing from monomer_properties.csv: " + ", ".join(missing)
        )

    for column in numeric_property_cols:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    rows: list[dict[str, float | str]] = []
    for sample_id, group in merged.groupby(sample_col, sort=False):
        total_weight = float(np.nansum(group["wt"]))
        row: dict[str, float | str] = {sample_col: sample_id, "wt_sum": total_weight}
        for column in numeric_property_cols:
            row[f"prop_wavg_{column}"] = float(
                np.nansum(group["wt"] * group[column]) / (total_weight + 1e-12)
            )
        rows.append(row)

    aggregated = pd.DataFrame(rows)
    feature_frame = formulations.merge(aggregated, on=sample_col, how="left")
    property_features = [f"prop_wavg_{column}" for column in numeric_property_cols]
    return feature_frame, sample_col, weight_cols, [*property_features, "wt_sum"]


def build_training_table(
    targets: pd.DataFrame,
    form_features: pd.DataFrame,
    sample_col: str,
    weight_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    tg_col = find_col(targets, ["tg", "tg_k", "tg.*k"], prefer="Tg_K")
    time_col = find_col(
        targets, ["cure.*time", "time.*s", "cure_time_s"], prefer="Cure_time_s"
    )
    target_sample_col = find_col(targets, ["sample", "id"], prefer="Sample_ID")
    if tg_col is None or time_col is None or target_sample_col is None:
        raise ValueError("Targets must contain Sample_ID, Cure_time_s, and Tg_K columns")

    frame = targets[[target_sample_col, time_col, tg_col]].rename(
        columns={target_sample_col: sample_col, time_col: "Cure_time_s", tg_col: "Tg_raw"}
    )
    frame = frame.merge(form_features, on=sample_col, how="left")
    frame["is_censored"] = (
        frame["Tg_raw"].astype(str).str.strip().str.lower().eq("out_of_range")
    )
    frame["Tg_K"] = pd.to_numeric(frame["Tg_raw"], errors="coerce")

    property_features = [
        column for column in form_features.columns if column.startswith("prop_wavg_")
    ]
    features = [*weight_cols, *property_features, "wt_sum", "Cure_time_s"]
    frame = frame.dropna(subset=features).copy()
    unknown_target = frame["Tg_K"].isna() & ~frame["is_censored"]
    if unknown_target.any():
        values = frame.loc[unknown_target, "Tg_raw"].astype(str).unique()
        raise ValueError(f"Unrecognized Tg target values: {values.tolist()}")
    if not (~frame["is_censored"] & frame["Tg_K"].notna()).any():
        raise ValueError("The targets file has no exact numeric Tg observations")
    return frame, features


def fit_catboost(
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    monotone_constraints: list[int],
    params: dict,
) -> CatBoostRegressor:
    pool = Pool(features, label=np.asarray(target, dtype=float).ravel())
    model = CatBoostRegressor(
        **params, monotone_constraints=monotone_constraints, allow_writing_files=False
    )
    model.fit(pool)
    return model


def train_model(
    frame: pd.DataFrame, features: list[str], config: Config, verbose: int | bool
) -> tuple[CatBoostRegressor, pd.DataFrame, float]:
    exact = frame[~frame["is_censored"] & frame["Tg_K"].notna()].copy()
    train_ids, validation_ids = train_test_split(
        exact.index.values, test_size=0.2, random_state=config.random_seed
    )
    monotone = [0] * len(features)
    monotone[features.index("Cure_time_s")] = 1
    params = {
        "loss_function": "RMSE",
        "iterations": config.catboost_iterations,
        "learning_rate": config.learning_rate,
        "depth": config.depth,
        "l2_leaf_reg": config.l2_leaf_reg,
        "random_seed": config.random_seed,
        "verbose": verbose,
    }

    model = fit_catboost(
        frame.loc[train_ids, features], frame.loc[train_ids, "Tg_K"], monotone, params
    )
    for iteration in range(config.em_iterations):
        mean = model.predict(frame[features])
        pseudo_target = frame["Tg_K"].copy()
        censored = frame["is_censored"].to_numpy()
        pseudo_target.loc[censored] = truncated_normal_mean_left(
            mean[censored], config.sigma_k, config.censoring_threshold_k
        )
        model = fit_catboost(
            frame[features].reset_index(drop=True),
            pseudo_target.reset_index(drop=True),
            monotone,
            params,
        )
        new_mean = model.predict(frame[features])
        delta = float(np.mean(np.abs(new_mean - mean)))
        print(f"EM iteration {iteration + 1}/{config.em_iterations}: mean |delta mu| = {delta:.4f} K")

    validation = frame.loc[validation_ids, ["Sample_ID", "Cure_time_s", "Tg_K"]].copy()
    validation["predicted_Tg_K"] = model.predict(frame.loc[validation_ids, features])
    validation["error_K"] = validation["predicted_Tg_K"] - validation["Tg_K"]
    mae = float(mean_absolute_error(validation["Tg_K"], validation["predicted_Tg_K"]))
    return model, validation, mae


def feature_importances(
    model: CatBoostRegressor,
    frame: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact = frame[~frame["is_censored"] & frame["Tg_K"].notna()]
    built_in = pd.DataFrame(
        {
            "feature": features,
            "importance": model.get_feature_importance(Pool(exact[features], exact["Tg_K"])),
        }
    ).sort_values("importance", ascending=False)
    permutation = permutation_importance(
        model,
        exact[features],
        exact["Tg_K"],
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=seed,
    )
    perm_frame = pd.DataFrame(
        {
            "feature": features,
            "mae_increase_K": permutation.importances_mean,
            "std_K": permutation.importances_std,
        }
    ).sort_values("mae_increase_K", ascending=False)
    return built_in, perm_frame


def plot_importance(data: pd.DataFrame, value: str, title: str, path: Path, color: str) -> None:
    ordered = data.sort_values(value)
    fig, axis = plt.subplots(figsize=(10, max(4, 0.25 * len(ordered))))
    axis.barh(ordered["feature"], ordered[value], color=color, alpha=0.9)
    axis.set_title(title)
    axis.set_xlabel(value.replace("_", " "))
    axis.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def prediction_times(frame: pd.DataFrame, config: Config) -> list[float]:
    times = [*map(float, config.prediction_times_s), float(frame["Cure_time_s"].max())]
    if len(set(times)) != 4:
        raise ValueError("Prediction times T1, T2, T3, and dataset maximum must be unique")
    return sorted(times)


def rank_training_formulations(
    model: CatBoostRegressor,
    frame: pd.DataFrame,
    form_features: pd.DataFrame,
    sample_col: str,
    weight_cols: list[str],
    features: list[str],
    times: list[float],
) -> pd.DataFrame:
    compositions = form_features.drop_duplicates(subset=[sample_col]).copy()
    grid = pd.concat(
        [compositions.assign(Cure_time_s=time) for time in times], ignore_index=True
    )
    grid["Tg_hat"] = model.predict(grid[features])
    wide = grid.pivot_table(
        index=sample_col, columns="Cure_time_s", values="Tg_hat", aggfunc="mean"
    ).reindex(times, axis=1)
    t1, _, t3, _ = times
    metrics = pd.DataFrame(
        {"Tg_T1": wide[t1], "d_early_T1_T3": wide[t3] - wide[t1]}
    )
    return (
        compositions.set_index(sample_col)[weight_cols]
        .join(metrics, how="inner")
        .reset_index()
        .sort_values(["Tg_T1", "d_early_T1_T3"])
        .reset_index(drop=True)
    )


def sample_component_weights(
    count: int, minimums: np.ndarray, maximums: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    if minimums.sum() >= 100.0:
        raise ValueError("The component minima must sum to less than 100 wt%")
    random_parts = rng.random((count, len(minimums)))
    random_parts /= random_parts.sum(axis=1, keepdims=True) + 1e-12
    weights = minimums[None, :] + (100.0 - minimums.sum()) * random_parts
    for _ in range(6):
        excess = np.maximum(0.0, weights - maximums[None, :]).sum(axis=1, keepdims=True)
        if np.all(excess <= 1e-9):
            break
        weights = np.minimum(weights, maximums[None, :])
        free = (weights < maximums[None, :] - 1e-12).astype(float)
        free_sum = free.sum(axis=1, keepdims=True)
        mask = free_sum[:, 0] > 0
        weights[mask] += excess[mask] * free[mask] / free_sum[mask]
    weights[:, -1] = 100.0 - weights[:, :-1].sum(axis=1)
    return weights


def weighted_properties(
    properties: pd.DataFrame, weights: np.ndarray, weight_cols: list[str]
) -> pd.DataFrame:
    numeric_cols = [
        column
        for column in properties.columns
        if column != "Component" and pd.api.types.is_numeric_dtype(properties[column])
    ]
    matrix = (
        properties.set_index("Component")[numeric_cols]
        .reindex(weight_cols)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    total = weights.sum(axis=1, keepdims=True) + 1e-12
    output = pd.DataFrame(
        (weights @ matrix) / total,
        columns=[f"prop_wavg_{column}" for column in numeric_cols],
    )
    output["wt_sum"] = weights.sum(axis=1)
    return output


def nearest_l1_distance(
    candidates: np.ndarray, training: np.ndarray, chunk_size: int = 20_000
) -> np.ndarray:
    """Compute nearest L1 distances in chunks to avoid a large temporary array."""
    output = np.empty(len(candidates), dtype=float)
    for start in range(0, len(candidates), chunk_size):
        stop = min(start + chunk_size, len(candidates))
        distances = np.abs(candidates[start:stop, None, :] - training[None, :, :]).sum(axis=2)
        output[start:stop] = distances.min(axis=1)
    return output


def generate_and_rank_candidates(
    model: CatBoostRegressor,
    properties: pd.DataFrame,
    form_features: pd.DataFrame,
    weight_cols: list[str],
    features: list[str],
    times: list[float],
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(weight_cols) < 5:
        raise ValueError("Expected four monomers plus an initiator column")
    optimized = weight_cols[:4]
    initiator = "TPOL"
    if initiator not in weight_cols:
        raise ValueError("The formulations file must contain the TPOL initiator column")

    rng = np.random.default_rng(config.random_seed)
    init_weight = rng.uniform(
        config.initiator_min_wt, config.initiator_max_wt, config.candidate_count
    )
    minimums = np.full(4, config.min_component_wt)
    maximums = np.full(4, config.max_component_wt)
    monomers = sample_component_weights(
        config.candidate_count, minimums, maximums, rng
    )
    monomers *= ((100.0 - init_weight) / 100.0)[:, None]

    candidate_weights = pd.DataFrame(0.0, index=range(config.candidate_count), columns=weight_cols)
    candidate_weights.loc[:, optimized] = monomers
    candidate_weights[initiator] = init_weight

    bounds_penalty = (
        np.maximum(0.0, minimums[None, :] - monomers).sum(axis=1)
        + np.maximum(0.0, monomers - maximums[None, :]).sum(axis=1)
    )
    distance_columns = [*optimized, initiator]
    nearest = nearest_l1_distance(
        candidate_weights[distance_columns].to_numpy(dtype=float),
        form_features[distance_columns].to_numpy(dtype=float),
    )
    close_penalty = np.maximum(0.0, config.min_l1_distance - nearest)
    penalty = (
        config.bounds_penalty_weight * bounds_penalty
        + config.closeness_penalty_weight * close_penalty
    )

    props = weighted_properties(
        properties, candidate_weights[weight_cols].to_numpy(dtype=float), weight_cols
    )
    base = pd.concat([candidate_weights.reset_index(drop=True), props], axis=1)
    predictions: dict[float, np.ndarray] = {}
    for time in times:
        design = base.copy()
        design["Cure_time_s"] = time
        predictions[time] = np.asarray(model.predict(design[features]), dtype=float)

    t1, t2, t3, t4 = times
    early_rise = predictions[t3] - predictions[t1]
    ranked = pd.DataFrame(
        {
            **{column: candidate_weights[column].to_numpy() for column in distance_columns},
            "Tg_T1": predictions[t1],
            "Tg_T2": predictions[t2],
            "Tg_T3": predictions[t3],
            "Tg_Tmax": predictions[t4],
            "d_early_T1_T3": early_rise,
            "nearest_training_L1": nearest,
            "penalty": penalty,
            "Tg_T1_penalized": predictions[t1] + penalty,
            "d_early_penalized": early_rise + penalty,
        }
    ).sort_values(["Tg_T1_penalized", "d_early_penalized"])
    ranked = ranked.reset_index(drop=True)
    display_cols = [
        *distance_columns,
        "Tg_T1",
        "Tg_T2",
        "Tg_T3",
        "Tg_Tmax",
        "d_early_T1_T3",
        "nearest_training_L1",
    ]
    return ranked, ranked[display_cols].head(config.top_n)


def plot_selection(
    training_rank: pd.DataFrame,
    top_candidates: pd.DataFrame,
    times: list[float],
    path: Path,
) -> None:
    x = training_rank["Tg_T1"].to_numpy()
    y = training_rank["d_early_T1_T3"].to_numpy()
    order = np.argsort(x)
    front_x: list[float] = []
    front_y: list[float] = []
    best_y = np.inf
    for x_value, y_value in zip(x[order], y[order]):
        if y_value <= best_y:
            front_x.append(x_value)
            front_y.append(y_value)
            best_y = y_value

    zone = training_rank.head(max(5, int(round(0.1 * len(training_rank)))))
    points = zone[["Tg_T1", "d_early_T1_T3"]].to_numpy()
    center = points.mean(axis=0)
    covariance = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order_e = np.argsort(eigenvalues)[::-1]
    axes = np.sqrt(5.991) * np.sqrt(np.maximum(eigenvalues[order_e], 1e-12))
    angles = np.linspace(0, 2.0 * np.pi, 200)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    ellipse = (circle * axes) @ eigenvectors[:, order_e].T + center

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(x, y, alpha=0.65, s=36, color="#5C95F8", label="Training formulations")
    axis.plot(front_x, front_y, lw=1.5, color="#0A58CC", label="Pareto front")
    axis.fill(ellipse[:, 0], ellipse[:, 1], color="#B6605F", alpha=0.10, label="Target region")
    axis.plot(ellipse[:, 0], ellipse[:, 1], lw=1.4, color="#E9502B")
    axis.scatter(
        top_candidates["Tg_T1"],
        top_candidates["d_early_T1_T3"],
        marker="D",
        s=70,
        color="#E9502B",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.82,
        label="Top ML candidates",
    )
    axis.set_xlabel(f"Tg ({times[0]:g} s), K")
    axis.set_ylabel(f"Delta Tg = Tg ({times[2]:g} s) - Tg ({times[0]:g} s), K")
    axis.set_title("Target region and top ML-ranked candidate formulations")
    axis.legend(loc="upper right", fontsize=9)
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.candidates <= 0 or args.top_n <= 0 or args.em_iterations < 0:
        raise ValueError("Candidates and top-n must be positive; EM iterations cannot be negative")
    config = Config(
        sigma_k=args.sigma_k,
        censoring_threshold_k=args.censoring_threshold_k,
        em_iterations=args.em_iterations,
        random_seed=args.seed,
        prediction_times_s=tuple(args.times),
        candidate_count=args.candidates,
        top_n=args.top_n,
        catboost_iterations=args.catboost_iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()

    formulations = read_csv_sc(data_dir / "formulations_wt_percent.csv")
    properties = read_csv_sc(data_dir / "monomer_properties.csv")
    targets = read_csv_sc(data_dir / "targets_Tg_conversion.csv")
    form_features, sample_col, weight_cols, _ = build_form_features(formulations, properties)
    frame, features = build_training_table(
        targets, form_features, sample_col, weight_cols
    )
    verbose: int | bool = False if args.quiet else 200
    model, validation, validation_mae = train_model(frame, features, config, verbose)
    times = prediction_times(frame, config)

    built_in, permutation = feature_importances(
        model, frame, features, config.random_seed
    )
    training_rank = rank_training_formulations(
        model, frame, form_features, sample_col, weight_cols, features, times
    )
    candidates, top_candidates = generate_and_rank_candidates(
        model, properties, form_features, weight_cols, features, times, config
    )

    model.save_model(output_dir / "tg_catboost_model.cbm")
    validation.to_csv(output_dir / "validation_predictions.csv", index=False)
    built_in.to_csv(output_dir / "feature_importance.csv", index=False)
    permutation.to_csv(output_dir / "permutation_importance.csv", index=False)
    training_rank.to_csv(output_dir / "training_formulation_ranking.csv", index=False)
    candidates.head(500).to_csv(output_dir / "candidate_top500.csv", index=False)
    top_candidates.to_csv(output_dir / "candidate_top.csv", index=False)
    plot_importance(
        built_in,
        "importance",
        "CatBoost feature importance",
        output_dir / "feature_importance.png",
        "#0A58CC",
    )
    plot_importance(
        permutation,
        "mae_increase_K",
        "Permutation impact on MAE",
        output_dir / "permutation_importance.png",
        "#E9502B",
    )
    plot_selection(
        training_rank, top_candidates, times, output_dir / "candidate_selection.png"
    )

    summary = {
        "status": "ok",
        "config": asdict(config),
        "input_files": {
            "formulations": str(data_dir / "formulations_wt_percent.csv"),
            "properties": str(data_dir / "monomer_properties.csv"),
            "targets": str(data_dir / "targets_Tg_conversion.csv"),
        },
        "training_rows": len(frame),
        "exact_rows": int((~frame["is_censored"] & frame["Tg_K"].notna()).sum()),
        "censored_rows": int(frame["is_censored"].sum()),
        "feature_count": len(features),
        "features": features,
        "prediction_times_s": times,
        "validation_mae_exact_only_K": validation_mae,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Completed. Exact-point validation MAE: {validation_mae:.4f} K")
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

