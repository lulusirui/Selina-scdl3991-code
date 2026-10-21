"""Run the dimension-by-weight MSE study, creating simulation data if needed."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from toad_study import (StudyConfig, build_encoder, evaluate_encoder, load_banks,
                        fit_preprocessing, make_group_pairs,
                        save_plot, train_encoder, transform)

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs" / "quantile_mse"
WEIGHTS = (0.1, 0.3, 0.5)
DIMENSIONS = (1, 2, 4, 8)
MIN_VALIDATION_ACCURACY = 0.80


def test_mse(saved, model, seed):
    """Compare cross-theta and same-theta pair MSE in test-standardized units."""
    summary = saved["summary"]
    scale = np.std(summary.reshape(-1, summary.shape[-1]), axis=0)
    scale = np.maximum(scale, 1e-6)
    left, right = make_group_pairs(model, seed + 800)
    k = summary.shape[1] // 2
    between = np.mean(((summary[left, :k] - summary[right, :k]) / scale) ** 2, axis=(1, 2))
    same = np.mean(((summary[left, :k] - summary[left, k:]) / scale) ** 2, axis=(1, 2))
    return float(between.mean()), float(same.mean())


def add_mse(row, prediction_path, model, seed):
    with np.load(prediction_path) as saved:
        between, same = test_mse(saved, model, seed)
    clean = {key: value for key, value in row.items() if "mmd" not in key}
    return {**clean, "between_parameter_mse": between,
            "same_parameter_mse": same}


def plot_results(table):
    colours = {"FFNN": "#0072B2", "CNN": "#D55E00"}
    d4 = table[table.dimension == 4]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for architecture, group in d4.groupby("architecture"):
        group = group.sort_values("mse_weight")
        style = dict(color=colours[architecture], marker="o", label=architecture)
        axes[0].plot(group.mse_weight, group.accuracy, **style)
        axes[1].plot(group.mse_weight, group.mean_positive_probe_r2, **style)
        axes[2].plot(group.mse_weight, group.between_parameter_mse, **style)
        axes[0].fill_between(group.mse_weight, group.accuracy_ci_low,
                             group.accuracy_ci_high, color=colours[architecture], alpha=.12)
    axes[0].axhline(1/3, color="black", ls=":", label="Chance")
    axes[0].set(ylabel="Held-out accuracy", ylim=(0.25, 1.02))
    axes[1].set(ylabel="Mean positive parameter-probe $R^2$")
    axes[2].set(ylabel="Different-parameter MSE")
    for ax in axes:
        ax.set_xlabel("MSE weight $\\lambda$")
        ax.legend(fontsize=8)
    save_plot(fig, OUTPUT, "01_mse_weight_tradeoff")

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
    for row_idx, architecture in enumerate(("FFNN", "CNN")):
        for col_idx, weight in enumerate((0.0, *WEIGHTS)):
            row = d4[(d4.architecture == architecture) & (d4.mse_weight == weight)].iloc[0]
            path = OUTPUT / "predictions" / f"{row.run}.npz"
            if not path.exists() and weight == 0:
                path = HERE / "outputs" / "quantile_mmd" / "predictions" / f"{row.run}.npz"
            with np.load(path) as saved:
                truth = np.repeat(saved["model"], saved["probabilities"].shape[1])
                chosen = saved["probabilities"].argmax(-1).ravel()
            cm = confusion_matrix(truth, chosen, labels=[0, 1, 2], normalize="true")
            ax = axes[row_idx, col_idx]
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                            color="white" if cm[i,j] > .55 else "black")
            ax.set(title=f"{architecture}, $\\lambda={weight:g}$", xticks=range(3), yticks=range(3),
                   xticklabels=["Random", "Nearest", "Distance"],
                   yticklabels=["Random", "Nearest", "Distance"],
                   xlabel="Predicted model", ylabel="True model")
    save_plot(fig, OUTPUT, "02_mse_confusion_matrices")

    weight_colours = {0.0: "#555555", 0.1: "#0072B2", 0.3: "#009E73", 0.5: "#D55E00"}
    for filename, column, ylabel in (
        ("03_accuracy_by_dimension", "accuracy", "Held-out classification accuracy"),
        ("04_probe_by_dimension", "mean_positive_probe_r2", "Mean positive parameter-probe $R^2$"),
        ("05_mse_by_dimension", "between_parameter_mse", "Different-parameter MSE"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, architecture in zip(axes, ("FFNN", "CNN")):
            subset = table[table.architecture == architecture]
            for weight, group in subset.groupby("mse_weight"):
                group = group.sort_values("dimension")
                ax.plot(group.dimension, group[column], "o-", color=weight_colours[weight],
                        label=f"$\\lambda={weight:g}$")
                if column == "accuracy":
                    ax.fill_between(group.dimension, group.accuracy_ci_low,
                                    group.accuracy_ci_high, color=weight_colours[weight], alpha=.08)
            ax.set(title=architecture, xlabel="Summary dimension $d$", xticks=DIMENSIONS)
            ax.legend(fontsize=8, loc="lower right" if column == "accuracy" else "upper left")
            if column == "accuracy":
                ax.axhline(1/3, color="black", ls=":", lw=1)
                ax.set_ylim(0.25, 1.02)
            else:
                ax.set_ylim(bottom=0)
        axes[0].set_ylabel(ylabel)
        if column != "accuracy":
            axes[0].set_ylim(0, table[column].max() * 1.12)
        save_plot(fig, OUTPUT, filename)


def main():
    config = StudyConfig(summary_dimensions=DIMENSIONS, mmd_weights=WEIGHTS)
    for folder in (OUTPUT, OUTPUT / "models", OUTPUT / "predictions", OUTPUT / "figures"):
        folder.mkdir(parents=True, exist_ok=True)
    old_output = HERE / "outputs" / "quantile_mmd"
    old_config = old_output / "run_config.json"
    old_table = old_output / "method_comparison.csv"
    old_cache = HERE / "data" / "quantile_mmd"
    cache_key = json.loads(old_config.read_text()).get("simulation_cache") if old_config.exists() else None
    reuse_old = bool(cache_key and old_table.exists() and all(
        (old_cache / f"{split}_{cache_key}.npz").exists() for split in ("train", "val", "test")))
    if reuse_old:
        banks = {}
        for split in ("train", "val", "test"):
            with np.load(old_cache / f"{split}_{cache_key}.npz") as saved:
                banks[split] = {name: saved[name] for name in saved.files}
            expected = sum(getattr(config, f"{split}_groups_by_model"))
            assert banks[split]["raw"].shape[0] == expected
        assert len(np.unique(np.concatenate([bank["group_id"] for bank in banks.values()]))) == sum(len(bank["group_id"]) for bank in banks.values())
        old = pd.read_csv(old_table)
    else:
        data_dir = HERE / "data" / "quantile_mse"
        data_dir.mkdir(parents=True, exist_ok=True)
        banks = load_banks(config, data_dir)
        old = None
    centre, scale = fit_preprocessing(banks["train"]["raw"])
    for bank in banks.values():
        bank["x"] = transform(bank["raw"], centre, scale)
    rows = []
    if (OUTPUT / "method_comparison.csv").exists():
        rows = pd.read_csv(OUTPUT / "method_comparison.csv").to_dict("records")
        rows = [{key: value for key, value in row.items()
                 if "mmd" not in key and key != "excess_mse"} for row in rows]
    completed = {row["run"] for row in rows}
    for architecture in ("FFNN", "CNN"):
        for dimension in DIMENSIONS:
            baseline = None
            if old is not None:
                candidates = old[(old.architecture == architecture) & (old.dimension == dimension)
                                 & (old.mmd_weight == 0)]
                if len(candidates) == 1:
                    candidate = candidates.iloc[0].to_dict()
                    prediction = old_output / "predictions" / f"{candidate['run']}.npz"
                    if prediction.exists():
                        baseline = candidate
            if baseline is not None and baseline["run"] not in completed:
                baseline["mse_weight"] = 0.0
                rows.append(add_mse(baseline, prediction, banks["test"]["model"], config.seed))
                pd.DataFrame(rows).to_csv(OUTPUT / "method_comparison.csv", index=False)
            weights = WEIGHTS if baseline is not None else (0.0, *WEIGHTS)
            for weight in weights:
                run = f"{architecture}_d{dimension}_mse{weight:g}"
                if run in completed:
                    continue
                keras.backend.clear_session()
                keras.utils.set_random_seed(config.seed + dimension + (100 if architecture == "CNN" else 0))
                encoder = build_encoder(architecture, dimension)
                history = train_encoder(encoder, banks, weight, config, OUTPUT, run,
                                        penalty="mse", min_validation_accuracy=MIN_VALIDATION_ACCURACY)
                metric, probes, _ = evaluate_encoder(encoder, banks, config, run, OUTPUT)
                eligible = history[history.val_accuracy >= MIN_VALIDATION_ACCURACY]
                best = eligible.loc[eligible.val_ce.idxmin()]
                metric.update(architecture=architecture, dimension=dimension, mse_weight=weight,
                              validation_ce=float(best.val_ce), validation_accuracy=float(best.val_accuracy),
                              epochs=len(history), trainable_parameters=encoder.count_params())
                metric = add_mse(metric, OUTPUT / "predictions" / f"{run}.npz",
                                  banks["test"]["model"], config.seed)
                rows.append(metric)
                mse_history = history.rename(columns={"mmd": "mse", "val_mmd": "val_mse"})
                mse_history.to_csv(OUTPUT / f"{run}_training_history.csv", index=False)
                pd.DataFrame(rows).to_csv(OUTPUT / "method_comparison.csv", index=False)
                probes.to_csv(OUTPUT / f"{run}_parameter_probe_r2.csv", index=False)
                print(f"Finished {run}: accuracy={metric['accuracy']:.3f}, probe R2={metric['mean_positive_probe_r2']:.3f}", flush=True)
    table = pd.DataFrame(rows).sort_values(["architecture", "dimension", "mse_weight"])
    assert len(table) == 2 * len(DIMENSIONS) * (1 + len(WEIGHTS))
    table.to_csv(OUTPUT / "method_comparison.csv", index=False)
    plot_results(table)
    (OUTPUT / "run_config.json").write_text(json.dumps({"base_config": config.__dict__,
        "penalty": "batch-standardized paired MSE", "checkpoint": "minimum validation CE subject to validation accuracy >= 0.80",
        "baseline": "reused quantile_mmd lambda=0 checkpoints when available; otherwise trained with MSE weight 0",
        "simulation_cache": cache_key if reuse_old else "generated in data/quantile_mse"}, indent=2))
    print(table[["architecture", "mse_weight", "accuracy", "mean_positive_probe_r2",
                 "between_parameter_mse", "same_parameter_mse"]].to_string(index=False))


if __name__ == "__main__":
    main()
