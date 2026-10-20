"""Dense d=4 MMD-weight sweep using the cached definitive simulation bank.

This script does not change the main 24-configuration experiment or its report.
It trains only previously untested weights and saves a separate collapse diagnostic.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

from toad_study import (StudyConfig, build_encoder, evaluate_encoder, fit_preprocessing,
                        load_banks, train_encoder, transform)


DIMENSION = 4
# The already completed main study supplies lambda = 0, 0.3 and 1.
NEW_WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80)


def main():
    study_dir = Path(__file__).resolve().parent
    main_output = study_dir / "outputs" / "quantile_mmd"
    output = study_dir / "outputs" / "lambda_sweep_d4"
    for folder in (output, output / "models", output / "predictions", output / "figures"):
        folder.mkdir(parents=True, exist_ok=True)

    config = StudyConfig()
    banks = load_banks(config, study_dir / "data" / "quantile_mmd")
    centre, scale = fit_preprocessing(banks["train"]["raw"])
    for bank in banks.values():
        bank["x"] = transform(bank["raw"], centre, scale)

    path = output / "lambda_sweep_metrics.csv"
    completed = pd.read_csv(path) if path.exists() else pd.DataFrame()
    rows = completed.to_dict("records")
    done = set(completed.run) if len(completed) else set()
    for architecture in ("FFNN", "CNN"):
        for weight in NEW_WEIGHTS:
            run = f"{architecture}_d4_lambda{weight:g}"
            if run in done:
                print(f"Already completed: {run}", flush=True)
                continue
            keras.backend.clear_session()
            keras.utils.set_random_seed(config.seed + DIMENSION + (100 if architecture == "CNN" else 0))
            encoder = build_encoder(architecture, DIMENSION)
            print(f"Starting {run}", flush=True)
            history = train_encoder(encoder, banks, weight, config, output, run)
            metric, _, _ = evaluate_encoder(encoder, banks, config, run, output)
            best_epoch = history.val_loss.idxmin()
            metric.update(architecture=architecture, dimension=DIMENSION, mmd_weight=weight,
                          validation_ce=float(history.loc[best_epoch, "val_ce"]),
                          validation_loss=float(history.loc[best_epoch, "val_loss"]),
                          epochs=len(history), trainable_parameters=encoder.count_params())
            rows.append(metric)
            pd.DataFrame(rows).sort_values(["architecture", "mmd_weight"]).to_csv(path, index=False)
            print(f"Finished {run}: accuracy={metric['accuracy']:.3f}, "
                  f"probe R²={metric['mean_positive_probe_r2']:.3f}", flush=True)

    sweep = pd.read_csv(path)
    base = pd.read_csv(main_output / "method_comparison.csv")
    base = base[(base.dimension == DIMENSION) & (base.architecture.isin(["FFNN", "CNN"]))]
    combined = (pd.concat([base, sweep], ignore_index=True)
                .drop_duplicates(["architecture", "mmd_weight"], keep="last")
                .sort_values(["architecture", "mmd_weight"]))
    combined.to_csv(output / "combined_d4_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for architecture, group in combined.groupby("architecture"):
        group = group.sort_values("mmd_weight")
        axes[0].plot(group.mmd_weight, group.accuracy, "o-", label=architecture)
        axes[1].plot(group.mmd_weight, group.mean_positive_probe_r2, "o-", label=architecture)
    axes[0].axhline(1/3, color="black", ls=":", label="Three-class chance")
    axes[0].set(xlabel="MMD weight λ", ylabel="Held-out classification accuracy", ylim=(0.25, 1.02),
                title="When does classification collapse?")
    axes[1].set(xlabel="MMD weight λ", ylabel="Mean positive parameter-probe R²",
                title="Parameter predictability at d=4")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "figures" / "d4_lambda_sweep.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / "figures" / "d4_lambda_sweep.pdf", bbox_inches="tight")
    print("Saved d=4 lambda-sweep plot.", flush=True)


if __name__ == "__main__":
    main()
