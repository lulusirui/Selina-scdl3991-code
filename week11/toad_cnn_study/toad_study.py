"""Quantile-input CNN/FFNN study: grouped conditional MMD and held-out probes.

Active workflow. Prior outputs in outputs/ belong to the archived raw-movement
experiment; new results go exclusively to outputs/quantile_mmd/.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-toad-quantiles")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, r2_score
from tensorflow import keras
from tensorflow.keras import layers

from toad_simulator import (LAGS, MODEL_NAMES, PARAMETER_NAMES, SUMMARY_NAMES,
                            draw_parameters, lag_summaries, simulate_toads)


@dataclass
class StudyConfig:
    seed: int = 3991
    num_toads: int = 66
    num_days: int = 63
    # 100,000 training parameter groups/datasets, split into exactly 50,000
    # same-model/different-parameter pairs: 16,667 + 16,667 + 16,666 pairs.
    train_groups_by_model: tuple = (33_334, 33_334, 33_332)
    val_groups_by_model: tuple = (96, 96, 96)
    test_groups_by_model: tuple = (128, 128, 128)
    train_replicates: int = 1
    evaluation_replicates: int = 12
    summary_dimensions: tuple = (1, 2, 4, 8)
    mmd_weights: tuple = (0.0, 0.3, 1.0)
    epochs: int = 35
    batch_pairs: int = 32
    patience: int = 7
    learning_rate: float = 0.001
    return_threshold: float = 10.0
    abc_accept: int = 30
    probe_trees: int = 100
    bootstrap_repeats: int = 300


def simulation_key(config):
    fields = ["seed", "num_toads", "num_days", "train_groups_by_model",
              "val_groups_by_model", "test_groups_by_model", "train_replicates",
              "evaluation_replicates", "return_threshold"]
    payload = {field: getattr(config, field) for field in fields}
    payload["simulator_sha256"] = hashlib.sha256(Path(__file__).with_name("toad_simulator.py").read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def simulate_group_bank(config, groups_by_model, split_id, replicates=None):
    """A group has one fixed theta and K independent trajectories; splits never share groups."""
    if isinstance(groups_by_model, int):
        groups_by_model = (groups_by_model,) * 3
    if len(groups_by_model) != 3:
        raise ValueError("groups_by_model must contain one count for each model")
    replicates = config.evaluation_replicates if replicates is None else replicates
    summaries, parameters, models, ids = [], [], [], []
    for m, group_count in enumerate(groups_by_model):
        for g in range(group_count):
            group_id = split_id * 1_000_000 + m * 100_000 + g
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, group_id]))
            theta = draw_parameters(m, rng)
            values = []
            for _ in range(replicates):
                positions = simulate_toads(m, theta, config.num_toads, config.num_days, rng)
                values.append(lag_summaries(positions, config.return_threshold))
            summaries.append(values)
            parameters.append(theta)
            models.append(m)
            ids.append(group_id)
    return dict(raw=np.asarray(summaries), theta=np.asarray(parameters),
                model=np.asarray(models), group_id=np.asarray(ids))


def load_banks(config, data_dir):
    key = simulation_key(config)
    banks = {}
    for split_id, split in enumerate(["train", "val", "test"], 1):
        path = data_dir / f"{split}_{key}.npz"
        if path.exists():
            with np.load(path) as saved:
                banks[split] = {name: saved[name] for name in saved.files}
            print(f"Loaded {split}: {banks[split]['raw'].shape}", flush=True)
        else:
            start = time.perf_counter()
            counts = getattr(config, f"{split}_groups_by_model")
            replicates = config.train_replicates if split == "train" else config.evaluation_replicates
            banks[split] = simulate_group_bank(config, counts, split_id, replicates)
            np.savez_compressed(path, **banks[split])
            print(f"Simulated {split}: {banks[split]['raw'].shape} in {time.perf_counter()-start:.1f}s", flush=True)
    all_ids = np.concatenate([bank["group_id"] for bank in banks.values()])
    assert len(np.unique(all_ids)) == len(all_ids), "Parameter groups overlap between splits"
    return banks


def fit_preprocessing(raw):
    flat = raw.reshape(-1, 48).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmedian(flat, axis=0)
    if not np.isfinite(centre).all():
        raise ValueError("A summary column is undefined in all training data")
    filled = np.where(np.isfinite(flat), flat, centre)
    scale = 1.4826 * np.median(np.abs(filled - centre), axis=0)
    fallback = np.std(filled, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
    return centre.astype(np.float32), scale.astype(np.float32)


def transform(raw, centre, scale):
    shape = raw.shape
    flat = raw.reshape(-1, 48)
    filled = np.where(np.isfinite(flat), flat, centre)
    return ((filled - centre) / scale).reshape(shape).astype(np.float32)


def make_group_pairs(models, seed):
    """Split each model bank in half: each group is used once in one fixed pair."""
    rng = np.random.default_rng(seed)
    left, right = [], []
    for m in range(3):
        ids = rng.permutation(np.flatnonzero(models == m))
        if len(ids) < 2 or len(ids) % 2:
            raise ValueError("Each model needs an even number of at least two groups")
        half = len(ids) // 2
        left.extend(ids[:half])
        right.extend(ids[half:])
    order = rng.permutation(len(left))
    left, right = np.asarray(left)[order], np.asarray(right)[order]
    assert np.all(left != right) and np.all(models[left] == models[right])
    assert len(np.unique(np.concatenate([left, right]))) == 2 * len(left)
    return left, right


def distribution_mmd(left, right):
    """Biased multiscale RBF MMD^2 for [parameter pair, replicate, dimension].

    A pooled spread bandwidth makes this invariant to uniform rescaling of a
    noncollapsed summary. Stop-gradient prevents a bandwidth shortcut. Finite K
    gives a positive sampling floor; same-theta control MMD is reported as well.
    Kernel distances are across DATASET vectors, never summary coordinates.
    """
    pooled = tf.concat([left, right], axis=1)
    centred = pooled - tf.reduce_mean(pooled, axis=1, keepdims=True)
    spread = tf.stop_gradient(tf.maximum(
        tf.reduce_mean(tf.reduce_sum(tf.square(centred), axis=-1), axis=1), 1e-12))

    def kernel(a, b):
        squared = tf.reduce_sum(tf.square(a[:, :, None, :] - b[:, None, :, :]), axis=-1)
        normalised = squared / spread[:, None, None]
        return tf.add_n([tf.exp(-normalised / (2 * factor**2)) for factor in (0.5, 1.0, 2.0)]) / 3

    return (tf.reduce_mean(kernel(left, left), axis=[1, 2])
            + tf.reduce_mean(kernel(right, right), axis=[1, 2])
            - 2 * tf.reduce_mean(kernel(left, right), axis=[1, 2]))


def singleton_kernel_distance(left, right):
    """Kernel distance for one dataset on each side of every training pair.

    Bandwidth is estimated across the whole minibatch. Estimating it separately
    from two singleton samples would make every normalized distance identical.
    """
    pooled = tf.concat([left, right], axis=0)
    centred = pooled - tf.reduce_mean(pooled, axis=0, keepdims=True)
    spread = tf.stop_gradient(tf.maximum(
        tf.reduce_mean(tf.reduce_sum(tf.square(centred), axis=-1)), 1e-12))
    squared = tf.reduce_sum(tf.square(left - right), axis=-1) / spread
    similarity = tf.add_n([tf.exp(-squared / (2 * factor**2))
                            for factor in (0.5, 1.0, 2.0)]) / 3
    return 2.0 - 2.0 * similarity


def standardized_pair_mse(left, right):
    """MSE of same-model pairs in pooled, batch-standardized summary units.

    The scale is differentiable: shrinking every coordinate cannot lower the
    penalty merely by changing units. The small floor handles constant inputs.
    """
    pooled = tf.concat([left, right], axis=0)
    scale = tf.sqrt(tf.math.reduce_variance(pooled, axis=0) + 1e-6)
    difference = (left - right) / scale
    return tf.reduce_mean(tf.square(difference))


def build_encoder(architecture, dimension):
    inputs = keras.Input((4, 12), name="lag_quantile_summaries")
    if architecture == "FFNN":
        hidden = layers.Flatten()(inputs)
        hidden = layers.Dense(64, activation="elu")(hidden)
        hidden = layers.Dense(32, activation="elu")(hidden)
    elif architecture == "CNN":
        # Count and median are scalars, not adjacent quantile gaps. Process
        # these separately; convolve ONLY over the ten ordered gap features.
        gaps = layers.Reshape((4, 10, 1))(inputs[:, :, 2:])
        gaps = layers.TimeDistributed(layers.Conv1D(12, 3, padding="same", activation="elu"))(gaps)
        gaps = layers.TimeDistributed(layers.Conv1D(16, 3, padding="same", activation="elu"))(gaps)
        gaps = layers.TimeDistributed(layers.Flatten())(gaps)
        combined = layers.Concatenate(axis=-1)([inputs[:, :, :2], gaps])
        hidden = layers.Flatten()(combined)  # lag identity/order must be retained
        hidden = layers.Dense(32, activation="elu")(hidden)
    else:
        raise ValueError(architecture)
    summary = layers.Dense(dimension, name="summary")(hidden)
    class_hidden = layers.Dense(24, activation="elu")(summary)
    probabilities = layers.Dense(3, activation="softmax", name="probabilities")(class_hidden)
    return keras.Model(inputs, {"summary": summary, "probabilities": probabilities},
                       name=f"{architecture.lower()}_d{dimension}")


def pair_objective(encoder, xa, xb, labels, weight, training, penalty="mmd"):
    n_pairs = tf.shape(xa)[0]
    k = tf.shape(xa)[1]
    output_a = encoder(tf.reshape(xa, (-1, 4, 12)), training=training)
    output_b = encoder(tf.reshape(xb, (-1, 4, 12)), training=training)
    repeated_labels = tf.repeat(labels, k)
    ce = 0.5 * (tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(repeated_labels, output_a["probabilities"]))
                + tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(repeated_labels, output_b["probabilities"])))
    sa = tf.reshape(output_a["summary"], (n_pairs, k, -1))
    sb = tf.reshape(output_b["summary"], (n_pairs, k, -1))
    if penalty == "mse":
        regularizer = standardized_pair_mse(tf.reshape(sa, (-1, tf.shape(sa)[-1])),
                                             tf.reshape(sb, (-1, tf.shape(sb)[-1])))
    elif penalty == "mmd" and xa.shape[1] == 1:
        regularizer = tf.reduce_mean(singleton_kernel_distance(sa[:, 0], sb[:, 0]))
    elif penalty == "mmd":
        regularizer = tf.reduce_mean(distribution_mmd(sa, sb))
    else:
        raise ValueError(f"Unknown penalty: {penalty}")
    acc = tf.reduce_mean(tf.cast(tf.argmax(output_a["probabilities"], axis=1, output_type=tf.int32) == repeated_labels, tf.float32))
    return ce + weight * regularizer, ce, regularizer, acc


def train_encoder(encoder, banks, weight, config, output, run_id, penalty="mmd",
                  min_validation_accuracy=None):
    optimiser = keras.optimizers.Adam(config.learning_rate, clipnorm=5.0)

    @tf.function(reduce_retracing=True)
    def step(xa, xb, labels):
        with tf.GradientTape() as tape:
            values = pair_objective(encoder, xa, xb, labels, weight, True, penalty)
        gradients = tape.gradient(values[0], encoder.trainable_variables)
        optimiser.apply_gradients(zip(gradients, encoder.trainable_variables))
        return values

    @tf.function(reduce_retracing=True)
    def evaluate(xa, xb, labels):
        return pair_objective(encoder, xa, xb, labels, weight, False, penalty)

    val_left, val_right = make_group_pairs(banks["val"]["model"], config.seed + 900)
    rows, best, wait = [], float("inf"), 0
    best_weights = encoder.get_weights()
    for epoch in range(config.epochs):
        # Clara's requested 50k-versus-50k pairing is fixed across epochs.
        left, right = make_group_pairs(banks["train"]["model"], config.seed)
        totals = np.zeros(4)
        for start in range(0, len(left), config.batch_pairs):
            a, b = left[start:start+config.batch_pairs], right[start:start+config.batch_pairs]
            bank = banks["train"]
            values = step(bank["x"][a], bank["x"][b], bank["model"][a].astype("int32"))
            totals += np.array([float(v) for v in values]) * len(a)
        totals /= len(left)
        val_totals = np.zeros(4)
        for start in range(0, len(val_left), config.batch_pairs):
            a, b = val_left[start:start+config.batch_pairs], val_right[start:start+config.batch_pairs]
            bank = banks["val"]
            values = evaluate(bank["x"][a], bank["x"][b], bank["model"][a].astype("int32"))
            val_totals += np.array([float(v) for v in values]) * len(a)
        val_totals /= len(val_left)
        row = {"run": run_id, "epoch": epoch+1}
        row.update(dict(zip(["loss", "ce", "mmd", "accuracy"], totals)))
        row.update(dict(zip(["val_loss", "val_ce", "val_mmd", "val_accuracy"], val_totals)))
        rows.append(row)
        selection_score = val_totals[1] if penalty == "mse" else val_totals[0]
        eligible = min_validation_accuracy is None or val_totals[3] >= min_validation_accuracy
        if eligible and selection_score < best - 1e-5:
            best, wait, best_weights = selection_score, 0, encoder.get_weights()
        elif np.isfinite(best):
            wait += 1
        if epoch == 0 or (epoch+1) % 10 == 0:
            print(f"  {run_id} epoch {epoch+1}: val CE={val_totals[1]:.3f}, {penalty.upper()}={val_totals[2]:.3f}, accuracy={val_totals[3]:.3f}", flush=True)
        if wait >= config.patience:
            break
    if not np.isfinite(best):
        raise RuntimeError(f"{run_id}: no checkpoint met the validation accuracy floor")
    encoder.set_weights(best_weights)
    encoder.save(output / "models" / f"{run_id}.keras")
    return pd.DataFrame(rows)


def predict_groups(encoder, bank):
    x = bank["x"]
    out = encoder.predict(x.reshape(-1, 4, 12), batch_size=256, verbose=0)
    return {key: value.reshape(*x.shape[:2], -1) for key, value in out.items()}


def bootstrap_accuracy(truth, predicted, config):
    rng = np.random.default_rng(config.seed+40)
    group_accuracy = np.mean(predicted == truth[:, None], axis=1)
    values = []
    for _ in range(config.bootstrap_repeats):
        ids = np.concatenate([rng.choice(np.flatnonzero(truth == m), size=np.sum(truth == m), replace=True) for m in range(3)])
        values.append(group_accuracy[ids].mean())
    return np.quantile(values, [0.025, 0.975])


def evaluate_encoder(encoder, banks, config, run_id, output):
    predictions = {split: predict_groups(encoder, bank) for split, bank in banks.items()}
    test = banks["test"]
    p = predictions["test"]["probabilities"]
    train_k = banks["train"]["raw"].shape[1]
    test_k = test["raw"].shape[1]
    truth = np.repeat(test["model"], test_k)
    chosen = p.argmax(axis=-1)
    low, high = bootstrap_accuracy(test["model"], chosen, config)
    group_left, group_right = make_group_pairs(test["model"], config.seed+800)
    mmd_rows = []
    k = test_k // 2
    s = predictions["test"]["summary"]
    # Equal-sized samples for different-theta and same-theta controls.
    between = distribution_mmd(tf.constant(s[group_left, :k]), tf.constant(s[group_right, :k])).numpy()
    within = distribution_mmd(tf.constant(s[group_left, :k]), tf.constant(s[group_left, k:2*k])).numpy()
    for i, a in enumerate(group_left):
        mmd_rows.append(dict(run=run_id, model=int(test["model"][a]),
                             group_a=int(test["group_id"][a]), group_b=int(test["group_id"][group_right[i]]),
                             between_parameter_mmd=float(between[i]), same_parameter_mmd=float(within[i])))

    probes = []
    for m in range(3):
        train_mask, test_mask = banks["train"]["model"] == m, test["model"] == m
        for j in range(4 if m == 2 else 3):
            xt = predictions["train"]["summary"][train_mask].reshape(-1, s.shape[-1])
            yt = np.repeat(banks["train"]["theta"][train_mask, j], train_k)
            xe = s[test_mask].reshape(-1, s.shape[-1])
            ye = np.repeat(test["theta"][test_mask, j], test_k)
            probe = ExtraTreesRegressor(n_estimators=config.probe_trees, min_samples_leaf=12,
                                        random_state=config.seed, n_jobs=-1)
            probe.fit(xt, yt)
            estimate = probe.predict(xe)
            probes.append(dict(run=run_id, model=m, parameter=PARAMETER_NAMES[j],
                               probe_r2=float(r2_score(ye, estimate))))
    probe_table = pd.DataFrame(probes)
    # Train-fitted RMS spread, independent of arbitrary coordinate magnitude.
    st = predictions["train"]["summary"].reshape(-1, s.shape[-1])
    total_variance = float(np.var(st, axis=0).sum())
    within_variance = float(np.mean([np.var(st[np.repeat(banks["train"]["model"], train_k) == m], axis=0).sum() for m in range(3)]))
    metrics = dict(run=run_id, accuracy=accuracy_score(truth, chosen.ravel()),
                   accuracy_ci_low=low, accuracy_ci_high=high,
                   macro_f1=f1_score(truth, chosen.ravel(), average="macro"),
                   log_loss=log_loss(truth, p.reshape(-1, 3), labels=[0, 1, 2]),
                   mean_probe_r2=probe_table.probe_r2.mean(),
                   mean_positive_probe_r2=probe_table.probe_r2.clip(lower=0).mean(),
                   between_parameter_mmd=float(between.mean()), same_parameter_mmd=float(within.mean()),
                   excess_mmd=float((between-within).mean()), summary_total_variance=total_variance,
                   within_total_variance_ratio=within_variance/max(total_variance, 1e-12))
    np.savez_compressed(output / "predictions" / f"{run_id}.npz",
                        summary=s, probabilities=p, model=test["model"], theta=test["theta"], group_id=test["group_id"])
    return metrics, probe_table, pd.DataFrame(mmd_rows)


def run_r_abc(banks, centre, data_dir, output, config):
    for split, filename in [("train", "training_bank.csv"), ("test", "test_summaries.csv")]:
        bank = banks[split]
        raw = bank["raw"].reshape(-1, 48)
        filled = np.where(np.isfinite(raw), raw, centre)
        frame = pd.DataFrame(filled, columns=[f"S{i+1}" for i in range(48)])
        k = bank["raw"].shape[1]
        frame.insert(0, "model" if split == "train" else "true_model", np.repeat(bank["model"], k))
        frame.insert(0, "group_id", np.repeat(bank["group_id"], k))
        if split == "test":
            frame.insert(0, "observation_id", np.arange(len(frame)))
        frame.to_csv(data_dir / filename, index=False)
    subprocess.run(["Rscript", str(Path(__file__).with_name("repository_style_abc.R")),
                    str(data_dir), str(output), str(config.abc_accept)], check=True)
    results = pd.read_csv(output / "abc_predictions.csv")
    results["group_id"] = np.repeat(banks["test"]["group_id"], banks["test"]["raw"].shape[1])
    results.to_csv(output / "abc_predictions.csv", index=False)
    return float(np.mean(results.true_model == results.predicted_model))


def save_plot(fig, output, filename):
    fig.tight_layout()
    fig.savefig(output / "figures" / f"{filename}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "figures" / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_plots(output, metrics, probes, history, config):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colours = {0.0: "#555555", 0.3: "#D55E00", 1.0: "#0072B2"}
    def colour(w): return colours.get(float(w), "#009E73")
    for filename, y, ylabel in [("01_accuracy_by_dimension", "accuracy", "Held-out accuracy"),
                                ("02_parameter_leakage_by_dimension", "mean_positive_probe_r2", "Mean positive parameter-probe R²")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, architecture in zip(axes, ["FFNN", "CNN"]):
            for weight, group in metrics[metrics.architecture == architecture].groupby("mmd_weight"):
                group = group.sort_values("dimension")
                ax.plot(group.dimension, group[y], "o-", color=colour(weight), label=f"λ = {weight:g}")
                if y == "accuracy":
                    ax.fill_between(group.dimension, group.accuracy_ci_low, group.accuracy_ci_high, color=colour(weight), alpha=.10)
            ax.set(title=architecture, xlabel="Learned summary dimension d", xticks=list(config.summary_dimensions), ylabel=ylabel)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8)
        save_plot(fig, output, filename)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, architecture in zip(axes, ["FFNN", "CNN"]):
        for weight, group in metrics[metrics.architecture == architecture].groupby("mmd_weight"):
            ax.scatter(group.mean_positive_probe_r2, group.accuracy, c=colour(weight), label=f"λ = {weight:g}", s=45)
            for _, row in group.iterrows():
                ax.annotate(f"d={int(row.dimension)}", (row.mean_positive_probe_r2, row.accuracy), xytext=(3, 4), textcoords="offset points", fontsize=8)
        ax.set(title=architecture, xlabel="Parameter-probe R² (positive part; lower is better)", ylabel="Accuracy (higher is better)")
        ax.legend(fontsize=8)
        ax.margins(.15)
    save_plot(fig, output, "03_accuracy_leakage_tradeoff")

    # Pick a representative by VALIDATION CE only; never optimise on test results.
    selection = metrics[metrics.architecture == "CNN"].sort_values(["validation_ce", "run"]).iloc[0]
    representative = selection.run
    match = metrics[(metrics.dimension == selection.dimension) & (metrics.mmd_weight == selection.mmd_weight) & (metrics.architecture == "FFNN")].iloc[0].run
    (output / "representative.json").write_text(json.dumps(dict(cnn=representative, ffnn=match,
        rule="CNN with minimum validation CE; FFNN at the same dimension and weight. Not selected on test scores."), indent=2))

    # Core classification diagnostic: the requested d=4 grid, with every
    # regularisation weight shown for each neural architecture.  The dormant
    # repository-style R script is deliberately not included in this figure.
    d4 = metrics[metrics.dimension == 4].sort_values(["architecture", "mmd_weight"])
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, row in zip(axes.ravel(), d4.itertuples()):
        run = row.run
        with np.load(output / "predictions" / f"{run}.npz") as data:
            repeats = data["probabilities"].shape[1]
            true = np.repeat(data["model"], repeats)
            predicted = data["probabilities"].argmax(-1).ravel()
        cm = confusion_matrix(true, predicted, labels=[0, 1, 2], normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3): ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", color="white" if cm[i,j] > .55 else "black")
        ax.set(title=f"{row.architecture}, $d=4$, $\\lambda={row.mmd_weight:g}$",
               xticks=range(3), yticks=range(3),
               xticklabels=["Random", "Nearest", "Distance"],
               yticklabels=["Random", "Nearest", "Distance"],
               xlabel="Predicted model", ylabel="True model")
    save_plot(fig, output, "04_d4_regularisation_confusion_matrices")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, architecture in zip(axes, ["FFNN", "CNN"]):
        subset = metrics[metrics.architecture == architecture]
        for w, group in subset.groupby("mmd_weight"):
            group = group.sort_values("dimension")
            ax.plot(group.dimension, group.between_parameter_mmd, "o-", c=colour(w), label=f"Different θ, λ={w:g}")
            ax.plot(group.dimension, group.same_parameter_mmd, ":", c=colour(w), label=f"Same θ control, λ={w:g}")
        ax.set(title=architecture, xlabel="Summary dimension d", ylabel="Held-out MMD² (matched sample sizes)", xticks=list(config.summary_dimensions))
        ax.legend(fontsize=7)
    save_plot(fig, output, "05_conditional_distribution_mmd")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, run in zip(axes, [match, representative]):
        subset = probes[probes.run == run]
        labels = [f"{['Random','Nearest','Distance'][r.model]} / {r.parameter}" for r in subset.itertuples()]
        ax.barh(labels, subset.probe_r2, color="#0072B2")
        ax.axvline(0, color="black", lw=.8)
        ax.set(title=run, xlabel="Held-out parameter-probe R²")
    save_plot(fig, output, "06_parameter_specific_probes")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, run in enumerate([match, representative]):
        h = history[history.run == run]
        axes[0, col].plot(h.epoch, h.ce, label="Train")
        axes[0, col].plot(h.epoch, h.val_ce, label="Validation")
        axes[1, col].plot(h.epoch, h.mmd, label="Train")
        axes[1, col].plot(h.epoch, h.val_mmd, label="Validation")
        axes[0, col].set(title=run, ylabel="Classification cross-entropy")
        axes[1, col].set(xlabel="Epoch", ylabel="Conditional MMD²")
        for row in range(2): axes[row, col].legend()
    save_plot(fig, output, "07_training_components")

    # The displayed coordinates are the actual coordinates (no parameter-fitted PCA).
    with np.load(output / "predictions" / f"{representative}.npz") as saved:
        summaries, theta, models = saved["summary"], saved["theta"], saved["model"]
    n_coords = min(2, summaries.shape[-1])
    fig, axes = plt.subplots(3, n_coords, figsize=(6*n_coords, 9), squeeze=False)
    for m in range(3):
        ids = np.flatnonzero(models == m)
        ordered = ids[np.argsort(theta[ids, 2])]
        chosen = ordered[[0, len(ordered)//2, -1]]
        for coordinate in range(n_coords):
            ax = axes[m, coordinate]
            for gid in chosen:
                values = np.sort(summaries[gid, :, coordinate])
                label = f"α={theta[gid,0]:.2f}, γ={theta[gid,1]:.0f}, p₀={theta[gid,2]:.2f}"
                if m == 2: label += f", d₀={theta[gid,3]:.0f}"
                ax.step(values, np.arange(1, len(values)+1)/len(values), where="post", label=label)
            ax.set(title=f"{MODEL_NAMES[m]} — coordinate {coordinate+1}", xlabel="Learned summary value", ylabel="Empirical CDF")
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"Fixed-parameter test groups: {representative}\nParameters vary jointly; overlap is descriptive, not proof of ancillarity", fontsize=11)
    save_plot(fig, output, "08_summary_distributions_at_fixed_parameters")

    # A direct view of the regularisation effect.  Unlike Figure 02, lambda is
    # now on the horizontal axis; chance-level classifiers are marked with x.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    dimension_colours = {1: "#0072B2", 2: "#D55E00", 4: "#009E73", 8: "#CC79A7"}
    for ax, architecture in zip(axes, ["FFNN", "CNN"]):
        subset = metrics[metrics.architecture == architecture]
        for dimension, group in subset.groupby("dimension"):
            group = group.sort_values("mmd_weight")
            ax.plot(group.mmd_weight, group.mean_positive_probe_r2, "o-",
                    color=dimension_colours[int(dimension)], label=f"d={int(dimension)}")
            collapsed = group[group.accuracy < 0.5]
            if len(collapsed):
                ax.scatter(collapsed.mmd_weight, collapsed.mean_positive_probe_r2,
                           marker="x", s=90, linewidths=2, color="black", zorder=5)
        ax.set(title=architecture, xlabel="MMD weight λ", xticks=list(config.mmd_weights))
        ax.legend(title="Summary dimension", fontsize=8)
    axes[0].set_ylabel("Final mean positive parameter-probe R² (lower is better)")
    fig.suptitle("Effect of regularisation on final parameter predictability\nBlack ×: classification accuracy below 0.5")
    save_plot(fig, output, "09_parameter_probe_by_regularisation")

    # Training progress for the two moderate-regularisation configurations used
    # in the main interpretation.  Parameter probes are post-training diagnostics,
    # so CE and accuracy—not probe R²—are the valid per-epoch quantities here.
    progress_runs = ["FFNN_d2_lambda0.3", "CNN_d4_lambda0.3"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, run in enumerate(progress_runs):
        h = history[history.run == run]
        axes[0, col].plot(h.epoch, h.ce, label="Train")
        axes[0, col].plot(h.epoch, h.val_ce, label="Validation")
        axes[0, col].set(title=run, ylabel="Classification cross-entropy")
        axes[1, col].plot(h.epoch, h.accuracy, label="Train")
        axes[1, col].plot(h.epoch, h.val_accuracy, label="Validation")
        axes[1, col].axhline(1/3, color="black", ls=":", lw=1, label="Chance")
        axes[1, col].set(xlabel="Epoch", ylabel="Classification accuracy", ylim=(0.25, 1.02))
        for row in range(2): axes[row, col].legend(fontsize=8)
    fig.suptitle("Training progress for moderate MMD regularisation")
    save_plot(fig, output, "10_regularised_training_progress")

    # Parameter-specific effect of moderate regularisation.  Negative R² is kept
    # because it is part of the diagnostic rather than clipped for this plot.
    probe_runs = ["FFNN_d2_lambda0", "FFNN_d2_lambda0.3",
                  "CNN_d4_lambda0", "CNN_d4_lambda0.3"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for ax, run in zip(axes.ravel(), probe_runs):
        subset = probes[probes.run == run]
        labels = [f"{MODEL_NAMES[int(r.model)]} / {r.parameter}" for r in subset.itertuples()]
        colours = ["#D55E00" if "lambda0.3" in run else "#555555"] * len(subset)
        ax.barh(labels, subset.probe_r2, color=colours)
        ax.axvline(0, color="black", lw=.8)
        ax.set(title=run, xlabel="Held-out parameter-probe R²")
    fig.suptitle("Which parameters remain predictable from the learned summary?")
    save_plot(fig, output, "11_parameter_specific_regularisation")


def run_study(study_dir, config=None):
    config = config or StudyConfig()
    if sum(config.train_groups_by_model) != 100_000:
        raise ValueError("The definitive design requires exactly 100,000 training datasets")
    if any(count % 2 for count in config.train_groups_by_model):
        raise ValueError("Each model's training count must be even for disjoint pairing")
    if config.evaluation_replicates < 4 or config.evaluation_replicates % 2:
        raise ValueError("Use an even number of evaluation replicates >= 4 for MMD controls")
    pair_count = sum(config.train_groups_by_model) // 2
    if pair_count != 50_000:
        raise ValueError("The definitive design requires exactly 50,000 training pairs")
    study_dir = Path(study_dir).resolve()
    output = study_dir / "outputs" / "quantile_mmd"
    data_dir = study_dir / "data" / "quantile_mmd"
    for directory in [output, data_dir, output / "figures", output / "models", output / "predictions"]:
        directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    banks = load_banks(config, data_dir)
    centre, scale = fit_preprocessing(banks["train"]["raw"])
    metadata = []
    for split, bank in banks.items():
        bank["x"] = transform(bank["raw"], centre, scale)
        for gid, model, theta in zip(bank["group_id"], bank["model"], bank["theta"]):
            metadata.append(dict(split=split, group_id=gid, model=model, **dict(zip(PARAMETER_NAMES, theta))))
    pd.DataFrame(metadata).to_csv(output / "parameter_groups.csv", index=False)
    pd.DataFrame({"feature": SUMMARY_NAMES, "training_median": centre, "training_scale": scale}).to_csv(output / "preprocessing.csv", index=False)
    np.savez(output / "preprocessing.npz", centre=centre, scale=scale)
    metrics, probes, distances, histories = [], [], [], []
    for architecture in ("FFNN", "CNN"):
        for dimension in config.summary_dimensions:
            for weight in config.mmd_weights:
                run_id = f"{architecture}_d{dimension}_lambda{weight:g}"
                keras.backend.clear_session()
                # Same initial weights and pairing schedule across lambda values.
                keras.utils.set_random_seed(config.seed + dimension + (100 if architecture == "CNN" else 0))
                encoder = build_encoder(architecture, dimension)
                h = train_encoder(encoder, banks, weight, config, output, run_id)
                metric, probe, distance = evaluate_encoder(encoder, banks, config, run_id, output)
                best_epoch = h.val_loss.idxmin()
                metric.update(architecture=architecture, dimension=dimension, mmd_weight=weight,
                              validation_ce=float(h.loc[best_epoch, "val_ce"]), validation_loss=float(h.loc[best_epoch, "val_loss"]),
                              epochs=len(h), trainable_parameters=encoder.count_params())
                metrics.append(metric)
                probes.append(probe)
                distances.append(distance)
                histories.append(h)
                pd.DataFrame(metrics).to_csv(output / "method_comparison.csv", index=False)
                pd.concat(probes).to_csv(output / "parameter_probe_r2.csv", index=False)
                pd.concat(distances).to_csv(output / "conditional_mmd.csv", index=False)
                pd.concat(histories).to_csv(output / "training_history.csv", index=False)
                print(f"Finished {run_id}: accuracy={metric['accuracy']:.3f}, probe R²={metric['mean_probe_r2']:.3f}", flush=True)
    metric_table, probe_table, history_table = pd.DataFrame(metrics), pd.concat(probes), pd.concat(histories)
    make_plots(output, metric_table, probe_table, history_table, config)
    report = asdict(config)
    report.update(training_pairs=pair_count,
                  training_simulations=banks["train"]["raw"].shape[0] * banks["train"]["raw"].shape[1],
                  total_simulations=sum(b["raw"].shape[0]*b["raw"].shape[1] for b in banks.values()),
                  simulation_cache=simulation_key(config),
                  elapsed_seconds=time.perf_counter()-start,
                  summary_definition="Per lag: return count, non-return median, 10 log adjacent raw-decile gaps",
                  split_unit="parameter group; no trajectory or group is shared",
                  mmd_estimator="biased multiscale RBF; pooled spread bandwidth; same-theta controls",
                  scope="single-seed 50,000-pair study; full 66-toad/63-day trajectories, simulated complete observations")
    (output / "run_config.json").write_text(json.dumps(report, indent=2))
    return metric_table
