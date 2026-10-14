from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "random_scale_summary_dimension_study.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md(r"""
    # Random-scale invariance and summary-dimension study

    This notebook extends the previous paired-statistic experiment in three steps:

    1. replace the fixed transformation $y\mapsto 2y$ with $y\mapsto ay$, where a new positive scale factor $a$ is drawn for every dataset;
    2. use the natural logarithm for preprocessing; and
    3. compare learned summary dimensions $d\in\{1,2,4,8\}$.

    The random scale prior is

    $$
    \log a\sim\operatorname{Uniform}(\log 0.1,\log 10),
    \qquad a\in[0.1,10].
    $$

    This prior is positive, bounded, and symmetric around the identity transformation on the log scale. It covers two orders of magnitude and includes both strong contractions and strong expansions. The value $a=0$ is excluded because $\log(0)$ is undefined and multiplying by zero would destroy all information in a dataset.

    Four configurations are trained. Every configuration uses random $a$ and `log` preprocessing; only the learned summary dimension changes across $d\in\{1,2,4,8\}$.

    All other main settings remain the same as in the previous experiment: the candidate models and priors, 100 observations per dataset, 3,000 datasets per model, the train/validation/test split, the shared CNN, the classification head receiving only the learned summary, $\lambda=0.5$, Adam, early stopping, and the post-hoc Extra Trees parameter probe.
    """),
    md(r"""
    ## 1. Imports and reproducibility

    The current experiment uses the same random seed and main training settings as the previous experiment. Running all four dimensions can take several times longer than training one model.
    """),
    code(r"""
    from pathlib import Path
    import os
    import random

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-scdl3991-random-scale")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
    from sklearn.model_selection import train_test_split
    from scipy.stats import spearmanr

    SEED = 3991
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    MODEL_NAMES = np.array(["Exponential", "Log-normal", "Gamma"])
    MODEL_COLOURS = ["#2A9D8F", "#E76F51", "#7A5195"]
    SAMPLE_SIZE = 100
    N_PER_MODEL = 3_000
    N_DATASETS = 3 * N_PER_MODEL
    EPOCHS = 40
    BATCH_SIZE = 256
    LAMBDA_STAT = 0.5
    SUMMARY_DIMENSIONS = [1, 2, 4, 8]
    EVALUATION_MULTIPLIERS = np.array(
        [0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0],
        dtype="float32",
    )
    EVALUATION_LABELS = ["0.1", "0.2", "0.5", "0.75", "1", "1.5", "2", "5", "10"]

    BASE_DIR = Path.cwd()
    if BASE_DIR.name != "week11" and (BASE_DIR / "week11").is_dir():
        BASE_DIR = BASE_DIR / "week11"
    OUTPUT_DIR = BASE_DIR / "outputs"
    MODEL_DIR = OUTPUT_DIR / "models"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    print("TensorFlow:", tf.__version__)
    print(f"Base simulations: {N_DATASETS:,}; observations per dataset: {SAMPLE_SIZE}")
    print("Output directory:", OUTPUT_DIR.resolve())
    """),
    md(r"""
    ## 2. Simulate the same three candidate models

    The data-generating models and parameter priors are unchanged:

    - $M_1$: Exponential(rate $\theta$), $\theta\sim\operatorname{Exp}(1)$;
    - $M_2$: Log-normal(location $\theta$, scale 1), $\theta\sim N(0,1)$; and
    - $M_3$: Gamma(shape 2, rate $\theta$), $\theta\sim\operatorname{Exp}(1)$.

    Each dataset is simulated once. Its paired view is formed deterministically from the same observations, so the pair never introduces a second source of sampling noise.
    """),
    code("""
    rng = np.random.default_rng(SEED)

    theta_exp = rng.exponential(scale=1.0, size=N_PER_MODEL)
    theta_logn = rng.normal(loc=0.0, scale=1.0, size=N_PER_MODEL)
    theta_gamma = rng.exponential(scale=1.0, size=N_PER_MODEL)

    exp_samples = rng.exponential(
        scale=1.0 / np.clip(theta_exp[:, None], 1e-6, None),
        size=(N_PER_MODEL, SAMPLE_SIZE),
    )
    logn_samples = rng.lognormal(
        mean=theta_logn[:, None], sigma=1.0,
        size=(N_PER_MODEL, SAMPLE_SIZE),
    )
    gamma_samples = rng.gamma(
        shape=2.0,
        scale=1.0 / np.clip(theta_gamma[:, None], 1e-6, None),
        size=(N_PER_MODEL, SAMPLE_SIZE),
    )

    X_raw = np.vstack([exp_samples, logn_samples, gamma_samples]).astype("float32")
    y_model = np.repeat(np.arange(3), N_PER_MODEL).astype("int32")
    y_onehot = keras.utils.to_categorical(y_model, num_classes=3).astype("float32")
    theta = np.concatenate([theta_exp, theta_logn, theta_gamma]).astype("float32")

    assert np.isfinite(X_raw).all()
    print("Raw datasets:", X_raw.shape)
    print("Class target:", y_onehot.shape)
    print("Parameter target retained for diagnostics only:", theta.shape)
    """),
    md(r"""
    ## 3. Draw a scale factor from a prior

    The previous experiment used the same fixed multiplier, $a=2$, for every dataset. In the current experiment, every dataset receives an independent scale factor:

    $$
    a_j=\exp(u_j),\qquad u_j\sim\operatorname{Uniform}(\log0.1,\log10).
    $$

    The paired raw data are $(y_j,a_jy_j)$. Conditional on $a_j$, the transformation changes the effective parameter to

    $$
    \theta_j^*=\theta_j/a_j
    $$

    for the Exponential and Gamma rate models, and to

    $$
    \theta_j^*=\theta_j+\log a_j
    $$

    for the Log-normal location model. The model family is unchanged.
    """),
    code("""
    A_MIN = 0.1
    A_MAX = 10.0
    N_SCALE_BINS = 8
    log_scale_factor = rng.uniform(
        np.log(A_MIN), np.log(A_MAX), size=N_DATASETS
    ).astype("float32")
    scale_factor = np.exp(log_scale_factor).astype("float32")

    print(pd.Series(scale_factor).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(scale_factor, bins=40, color="#457B9D", alpha=0.85)
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Scale factor a", ylabel="Datasets", title="Prior draws on the original scale")
    axes[1].hist(log_scale_factor, bins=40, color="#E9C46A", alpha=0.90)
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="log(a)", ylabel="Datasets", title="Uniform prior on the log scale")
    fig.suptitle("Distribution of sampled scale factors (design check only)", fontsize=14, weight="bold")
    fig.tight_layout()
    plt.show()
    """),
    md("""
    ## 4. Split the base simulations before constructing model inputs

    The 64%/16%/20% stratified split is unchanged. One base simulation and every transformed version of it always remain in the same subset.
    """),
    code("""
    all_idx = np.arange(N_DATASETS)
    train_val_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, random_state=SEED, stratify=y_model
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.20,
        random_state=SEED + 1,
        stratify=y_model[train_val_idx],
    )

    print("Train / validation / test:", len(train_idx), len(val_idx), len(test_idx))
    """),
    md(r"""
    ## 5. Log preprocessing

    The current experiment uses

    $$x=\log y.$$

    All three candidate models generate strictly positive observations. A very small lower clip is included only for numerical safety. Under log preprocessing, random scaling becomes an additive shift:

    $$
    \log(ay)=\log y+\log a.
    $$

    No per-dataset standardization is applied, because it could remove the scale change before the network learns from it.
    """),
    code("""
    def log_preprocess(values):
        values = np.asarray(values, dtype="float32")
        transformed = np.log(np.clip(values, 1e-8, None))
        return transformed[..., None].astype("float32")


    def make_views():
        paired_raw = X_raw * scale_factor[:, None]
        original = log_preprocess(X_raw)
        transformed = log_preprocess(paired_raw)
        assert np.isfinite(original).all() and np.isfinite(transformed).all()
        return original, transformed


    X_original, X_transformed = make_views()
    print("Original log input:", X_original.shape)
    print("Randomly scaled log input:", X_transformed.shape)
    print("Maximum numerical error in log(ay) = log(y) + log(a):",
          float(np.max(np.abs(
              X_transformed[..., 0]
              - X_original[..., 0]
              - log_scale_factor[:, None]
          ))))
    """),
    md(r"""
    ## 6. Shared CNN with a configurable summary dimension

    The feature extractor from the previous experiment is retained:

    $$
    100\rightarrow\operatorname{Conv1D}(32)\rightarrow\operatorname{MaxPool}
    \rightarrow\operatorname{Conv1D}(64)\rightarrow\operatorname{GlobalAveragePool}\rightarrow64.
    $$

    Only the final summary layer changes from $64\to1$ to $64\to d$. The classifier still receives **only** the learned summary. Therefore, every dimension must balance model classification against paired invariance; the classifier cannot bypass the bottleneck.

    For $s_d(y)\in\mathbb{R}^d$, the objective is

    $$
    L=\frac{CE(M,\widehat M_y)+CE(M,\widehat M_{ay})}{2}
    +0.5\frac{1}{d}\lVert s_d(y)-s_d(ay)\rVert_2^2.
    $$

    The mean over coordinates keeps the similarity loss comparable as $d$ changes.
    """),
    code("""
    def build_models(summary_dim, seed):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)

        inputs = keras.Input(shape=(SAMPLE_SIZE, 1), name="dataset")
        x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu", name="conv_1")(inputs)
        x = layers.MaxPooling1D(pool_size=2, name="max_pool")(x)
        x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu", name="conv_2")(x)
        x = layers.GlobalAveragePooling1D(name="global_average_pool")(x)
        embedding = layers.Dense(64, activation="relu", name="learned_embedding")(x)
        statistic = layers.Dense(summary_dim, activation="linear", name="statistic")(embedding)

        class_hidden = layers.Dense(24, activation="elu", name="class_hidden")(statistic)
        class_output = layers.Dense(3, activation="softmax", name="class_output")(class_hidden)
        shared_cnn = keras.Model(
            inputs,
            {"class_output": class_output, "statistic": statistic},
            name=f"shared_cnn_d{summary_dim}",
        )

        original_input = keras.Input(shape=(SAMPLE_SIZE, 1), name="original_dataset")
        transformed_input = keras.Input(shape=(SAMPLE_SIZE, 1), name="transformed_dataset")
        original_outputs = shared_cnn(original_input)
        transformed_outputs = shared_cnn(transformed_input)

        class_original = layers.Activation("linear", name="class_original")(
            original_outputs["class_output"]
        )
        class_transformed = layers.Activation("linear", name="class_transformed")(
            transformed_outputs["class_output"]
        )
        stat_difference = layers.Subtract(name="stat_difference")([
            original_outputs["statistic"], transformed_outputs["statistic"]
        ])

        paired_model = keras.Model(
            inputs={
                "original_dataset": original_input,
                "transformed_dataset": transformed_input,
            },
            outputs={
                "class_original": class_original,
                "class_transformed": class_transformed,
                "stat_difference": stat_difference,
            },
            name=f"paired_invariant_cnn_d{summary_dim}",
        )
        paired_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=5.0),
            loss={
                "class_original": "categorical_crossentropy",
                "class_transformed": "categorical_crossentropy",
                "stat_difference": "mse",
            },
            loss_weights={
                "class_original": 0.5,
                "class_transformed": 0.5,
                "stat_difference": LAMBDA_STAT,
            },
            metrics={
                "class_original": ["accuracy"],
                "class_transformed": ["accuracy"],
                "stat_difference": ["mse"],
            },
        )
        return shared_cnn, paired_model
    """),
    md("""
    ## 7. Define the summary-dimension comparison

    Random scaling and log preprocessing are fixed throughout. Only the learned summary dimension changes.
    """),
    code("""
    CONFIGURATIONS = [
        {"name": "random_a_log_d1", "summary_dim": 1},
        {"name": "random_a_log_d2", "summary_dim": 2},
        {"name": "random_a_log_d4", "summary_dim": 4},
        {"name": "random_a_log_d8", "summary_dim": 8},
    ]

    pd.DataFrame(CONFIGURATIONS)
    """),
    md("""
    ## 8. Train and evaluate every configuration

    Each model sees the same raw simulations and the same train/validation/test indices. The random-scale configurations also use the same draw of $a$ for every summary dimension. This makes the comparisons paired and reduces avoidable experimental variation.

    The parameter $\theta$ is never a training target. It is used only after training in a fresh Extra Trees probe. Lower probe $R^2$ means less parameter information can be recovered from the learned summary.
    """),
    code("""
    comparison_rows = []
    probe_rows = []
    scale_bin_rows = []
    history_frames = []
    predictions = {}

    for config_id, config in enumerate(CONFIGURATIONS):
        name = config["name"]
        summary_dim = config["summary_dim"]
        print("\\n" + "=" * 78)
        print(f"Training {name}")

        shared_cnn, paired_model = build_models(summary_dim, SEED + config_id)

        def paired_inputs(idx):
            return {
                "original_dataset": X_original[idx],
                "transformed_dataset": X_transformed[idx],
            }

        def paired_targets(idx):
            return {
                "class_original": y_onehot[idx],
                "class_transformed": y_onehot[idx],
                "stat_difference": np.zeros((len(idx), summary_dim), dtype="float32"),
            }

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
            ),
        ]
        history = paired_model.fit(
            paired_inputs(train_idx),
            paired_targets(train_idx),
            validation_data=(paired_inputs(val_idx), paired_targets(val_idx)),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=2,
        )

        history_df = pd.DataFrame(history.history)
        history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
        history_df.insert(0, "configuration", name)
        history_frames.append(history_df)

        pred_train = shared_cnn.predict(
            X_original[train_idx], batch_size=BATCH_SIZE, verbose=0
        )
        pred_original = shared_cnn.predict(
            X_original[test_idx], batch_size=BATCH_SIZE, verbose=0
        )
        pred_transformed = shared_cnn.predict(
            X_transformed[test_idx], batch_size=BATCH_SIZE, verbose=0
        )

        prob_original = pred_original["class_output"]
        prob_transformed = pred_transformed["class_output"]
        stat_train = pred_train["statistic"]
        stat_original = pred_original["statistic"]
        stat_transformed = pred_transformed["statistic"]
        stat_difference = stat_transformed - stat_original

        class_original = prob_original.argmax(axis=1)
        class_transformed = prob_transformed.argmax(axis=1)
        true_test = y_model[test_idx]
        accuracy_original = accuracy_score(true_test, class_original)
        accuracy_transformed = accuracy_score(true_test, class_transformed)
        paired_mse = float(np.mean(stat_difference ** 2))
        paired_rmse = float(np.sqrt(paired_mse))
        paired_mae = float(np.mean(np.abs(stat_difference)))

        per_row_original_norm = np.linalg.norm(stat_original, axis=1)
        per_row_difference_norm = np.linalg.norm(stat_difference, axis=1)
        relative_change = float(np.mean(
            per_row_difference_norm / np.maximum(per_row_original_norm, 1e-6)
        ))

        comparison_rows.append({
            **config,
            "epochs_trained": len(history_df),
            "accuracy_original": accuracy_original,
            "accuracy_transformed": accuracy_transformed,
            "paired_mse_per_coordinate": paired_mse,
            "paired_rmse_per_coordinate": paired_rmse,
            "paired_mae_per_coordinate": paired_mae,
            "mean_relative_vector_change": relative_change,
        })

        for model_id, model_name in enumerate(MODEL_NAMES):
            train_mask = y_model[train_idx] == model_id
            test_mask = true_test == model_id
            probe = ExtraTreesRegressor(
                n_estimators=250,
                min_samples_leaf=8,
                random_state=SEED,
                n_jobs=-1,
            )
            probe.fit(stat_train[train_mask], theta[train_idx][train_mask])
            theta_pred = probe.predict(stat_original[test_mask])

            component_rhos = [
                spearmanr(
                    theta[test_idx][test_mask], stat_original[test_mask, component]
                ).statistic
                for component in range(summary_dim)
            ]
            probe_rows.append({
                "configuration": name,
                "summary_dim": summary_dim,
                "model": model_name,
                "theta_probe_r2": float(r2_score(theta[test_idx][test_mask], theta_pred)),
                "max_absolute_component_spearman": float(np.nanmax(np.abs(component_rhos))),
            })

        test_log_a = log_scale_factor[test_idx]
        scale_bins = pd.qcut(
            test_log_a, q=N_SCALE_BINS, labels=False, duplicates="drop"
        )
        for bin_id in sorted(np.unique(scale_bins)):
            bin_mask = np.asarray(scale_bins == bin_id)
            scale_bin_rows.append({
                "configuration": name,
                "summary_dim": summary_dim,
                "scale_bin": int(bin_id),
                "mean_a": float(np.mean(scale_factor[test_idx][bin_mask])),
                "paired_rmse_per_coordinate": float(np.sqrt(
                    np.mean(stat_difference[bin_mask] ** 2)
                )),
                "transformed_accuracy": float(accuracy_score(
                    true_test[bin_mask], class_transformed[bin_mask]
                )),
            })

        predictions[name] = {
            "stat_original": stat_original,
            "stat_transformed": stat_transformed,
            "class_original": class_original,
            "class_transformed": class_transformed,
            "factors": scale_factor[test_idx],
        }
        shared_cnn.save(MODEL_DIR / f"{name}.keras")
        print(
            f"accuracy: {accuracy_original:.3f} / {accuracy_transformed:.3f}; "
            f"paired RMSE per coordinate: {paired_rmse:.4f}"
        )

    comparison_metrics = pd.DataFrame(comparison_rows)
    probe_metrics = pd.DataFrame(probe_rows)
    scale_bin_metrics = pd.DataFrame(scale_bin_rows)
    training_history = pd.concat(history_frames, ignore_index=True)

    comparison_metrics.to_csv(OUTPUT_DIR / "configuration_comparison.csv", index=False)
    probe_metrics.to_csv(OUTPUT_DIR / "theta_probe_by_model.csv", index=False)
    scale_bin_metrics.to_csv(OUTPUT_DIR / "scale_bin_metrics.csv", index=False)
    training_history.to_csv(OUTPUT_DIR / "training_history_all_configurations.csv", index=False)

    display(comparison_metrics.round(4))
    """),
    md(r"""
    ## 9. Evaluate exact multiplier values

    Random draws from a continuous distribution almost never equal exactly $a=1$. To make the multiplier comparison unambiguous, each trained model is evaluated again on the same held-out datasets using the fixed grid

    $$
    a\in\{0.1,0.2,0.5,0.75,1,1.5,2,5,10\}.
    $$

    This grid explicitly contains $a=1$. At that value, $ay=y$, so the summary discrepancy must be zero up to floating-point precision.
    """),
    code("""
    exact_multiplier_rows = []
    X_test_original = log_preprocess(X_raw[test_idx])
    true_test = y_model[test_idx]

    for config in CONFIGURATIONS:
        name = config["name"]
        summary_dim = config["summary_dim"]
        evaluation_model = keras.models.load_model(
            MODEL_DIR / f"{name}.keras", compile=False
        )
        original_prediction = evaluation_model.predict(
            X_test_original, batch_size=BATCH_SIZE, verbose=0
        )
        original_statistic = original_prediction["statistic"]

        for multiplier in EVALUATION_MULTIPLIERS:
            X_test_scaled = log_preprocess(X_raw[test_idx] * multiplier)
            scaled_prediction = evaluation_model.predict(
                X_test_scaled, batch_size=BATCH_SIZE, verbose=0
            )
            scaled_statistic = scaled_prediction["statistic"]
            difference = scaled_statistic - original_statistic
            exact_multiplier_rows.append({
                "configuration": name,
                "summary_dim": summary_dim,
                "multiplier_a": float(multiplier),
                "paired_mse_per_coordinate": float(np.mean(difference ** 2)),
                "paired_rmse_per_coordinate": float(np.sqrt(np.mean(difference ** 2))),
                "paired_mae_per_coordinate": float(np.mean(np.abs(difference))),
                "classification_accuracy": float(accuracy_score(
                    true_test, scaled_prediction["class_output"].argmax(axis=1)
                )),
            })
        del evaluation_model

    exact_multiplier_metrics = pd.DataFrame(exact_multiplier_rows)
    exact_multiplier_metrics.to_csv(
        OUTPUT_DIR / "exact_multiplier_metrics.csv", index=False
    )

    identity_check = exact_multiplier_metrics.query("multiplier_a == 1.0")
    display(identity_check[[
        "summary_dim", "multiplier_a", "paired_mse_per_coordinate",
        "paired_rmse_per_coordinate", "classification_accuracy",
    ]].round(10))
    """),
    md("""
    ## 10. Headline comparison across summary dimensions

    The first plot compares classification, paired invariance, and parameter leakage across every configuration. Higher accuracy is better. Lower paired RMSE and lower probe $R^2$ are better for the invariance objective.

    Every bar refers to random-$a$ pairs with log preprocessing, so differences are attributable to summary dimension and training variation.
    """),
    code("""
    plot_metrics = comparison_metrics.copy()
    short_labels = [
        "random a\\nlog d=1",
        "random a\\nlog d=2",
        "random a\\nlog d=4",
        "random a\\nlog d=8",
    ]
    x = np.arange(len(plot_metrics))
    width = 0.36

    mean_probe = probe_metrics.groupby("configuration", sort=False)["theta_probe_r2"].mean()
    mean_probe = mean_probe.reindex(plot_metrics["name"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].bar(x - width / 2, plot_metrics["accuracy_original"], width, label="Original y")
    axes[0, 0].bar(x + width / 2, plot_metrics["accuracy_transformed"], width, label="Transformed ay")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Held-out accuracy")
    axes[0, 0].set_title("Model information retained")
    axes[0, 0].legend()

    axes[0, 1].bar(x, plot_metrics["paired_rmse_per_coordinate"], color="#457B9D")
    axes[0, 1].set_ylabel("Paired RMSE per coordinate")
    axes[0, 1].set_title("Summary invariance (lower is better)")

    axes[1, 0].bar(x, mean_probe, color="#E76F51")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_ylabel(r"Mean parameter-probe $R^2$")
    axes[1, 0].set_title("Parameter leakage (lower is better)")

    axes[1, 1].bar(x, plot_metrics["epochs_trained"], color="#7A5195")
    axes[1, 1].set_ylabel("Epochs before early stopping")
    axes[1, 1].set_title("Training duration")

    for ax in axes.ravel():
        ax.set_xticks(x, short_labels, rotation=20, ha="right")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Comparison across summary dimensions", fontsize=16, weight="bold")
    fig.tight_layout()
    plt.show()
    """),
    md("""
    ## 11. Effect of summary dimension

    These plots use only the random-scale, log-preprocessed models. They show whether extra summary coordinates improve classification, weaken paired invariance, or retain more nuisance-parameter information.
    """),
    code("""
    dimension_metrics = comparison_metrics.sort_values("summary_dim")
    dimension_probe = probe_metrics.copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].plot(dimension_metrics["summary_dim"], dimension_metrics["accuracy_original"],
                 marker="o", linewidth=2, label="Original y")
    axes[0].plot(dimension_metrics["summary_dim"], dimension_metrics["accuracy_transformed"],
                 marker="o", linewidth=2, label="Transformed ay")
    axes[0].set(xlabel="Summary dimension d", ylabel="Held-out accuracy",
                title="Classification")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    axes[1].plot(dimension_metrics["summary_dim"],
                 dimension_metrics["paired_rmse_per_coordinate"],
                 marker="o", linewidth=2, color="#457B9D")
    axes[1].set(xlabel="Summary dimension d", ylabel="Paired RMSE per coordinate",
                title="Random-scale invariance")

    for model_name, colour in zip(MODEL_NAMES, MODEL_COLOURS):
        subset = dimension_probe[dimension_probe["model"] == model_name].sort_values("summary_dim")
        axes[2].plot(subset["summary_dim"], subset["theta_probe_r2"],
                     marker="o", linewidth=2, color=colour, label=model_name)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set(xlabel="Summary dimension d", ylabel=r"Parameter-probe $R^2$",
                title="Parameter information")
    axes[2].legend()

    for ax in axes:
        ax.set_xticks(SUMMARY_DIMENSIONS)
    fig.suptitle("What changes when the learned summary has more dimensions?",
                 fontsize=15, weight="bold")
    fig.tight_layout()
    plt.show()
    """),
    md("""
    ## 12. Performance at exact multiplier values

    A model can perform well on average while struggling near the edges of the scale prior. Each point below is evaluated at the exact multiplier printed on the horizontal axis. The value $a=1$ is included explicitly.
    """),
    code("""
    exact_scale = exact_multiplier_metrics.copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for dimension in SUMMARY_DIMENSIONS:
        subset = exact_scale[
            exact_scale["summary_dim"] == dimension
        ].sort_values("multiplier_a")
        axes[0].plot(subset["multiplier_a"], subset["paired_rmse_per_coordinate"],
                     marker="o", linewidth=2, label=f"d={dimension}")
        axes[1].plot(subset["multiplier_a"], subset["classification_accuracy"],
                     marker="o", linewidth=2, label=f"d={dimension}")

    axes[0].set(xlabel="Exact scale factor a", ylabel="Paired RMSE per coordinate",
                title="Invariance at exact multipliers")
    axes[1].set(xlabel="Exact scale factor a", ylabel="Scaled-view accuracy",
                title="Classification at exact multipliers")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(EVALUATION_MULTIPLIERS)
        ax.set_xticklabels(EVALUATION_LABELS)
        ax.minorticks_off()
        ax.axvline(1.0, color="black", linestyle="--", linewidth=0.9)
        ax.legend(title="Summary dimension")
    fig.suptitle("Performance at specified multiplier values",
                 fontsize=15, weight="bold")
    fig.tight_layout()
    plt.show()
    """),
    md(r"""
    ## 13. Direct invariance diagnostics across multiplier values

    The previous histogram only describes how $a$ was sampled. The following plots test the learned summaries directly. For each held-out pair and summary dimension, the per-dataset discrepancy is

    $$
    e_a=\sqrt{\frac{1}{d}\lVert s_d(ay)-s_d(y)\rVert_2^2}.
    $$

    Values close to zero indicate stronger invariance. The coloured points use the continuously sampled test multipliers. The black curve evaluates the same held-out datasets at the exact multiplier values printed on the horizontal axis.
    """),
    code("""
    test_a = scale_factor[test_idx]
    true_test = y_model[test_idx]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharex=True)

    for ax, dimension in zip(axes.ravel(), SUMMARY_DIMENSIONS):
        values = predictions[f"random_a_log_d{dimension}"]
        difference = values["stat_transformed"] - values["stat_original"]
        per_dataset_error = np.sqrt(np.mean(difference ** 2, axis=1))

        for model_id, (model_name, colour) in enumerate(zip(MODEL_NAMES, MODEL_COLOURS)):
            mask = true_test == model_id
            ax.scatter(test_a[mask], per_dataset_error[mask], s=11, alpha=0.20,
                       color=colour, label=model_name)

        exact_subset = exact_multiplier_metrics[
            exact_multiplier_metrics["summary_dim"] == dimension
        ].sort_values("multiplier_a")
        ax.plot(exact_subset["multiplier_a"],
                exact_subset["paired_rmse_per_coordinate"],
                color="black", marker="o", linewidth=2.2,
                label="Exact multiplier evaluation")
        ax.set_xscale("log")
        ax.set_xticks(EVALUATION_MULTIPLIERS)
        ax.set_xticklabels(EVALUATION_LABELS)
        ax.minorticks_off()
        ax.axvline(1.0, color="black", linestyle="--", linewidth=0.9)
        ax.set(title=f"Summary dimension d={dimension}",
               xlabel="Scale factor a", ylabel="Per-dataset summary discrepancy")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True)
    fig.suptitle("Does summary invariance change with the multiplier?",
                 fontsize=15, weight="bold", y=1.01)
    fig.tight_layout()
    plt.show()
    """),
    md(r"""
    For the one-dimensional summary, paired agreement can also be shown directly by plotting $s(y)$ against $s(ay)$. Points on the diagonal have identical summaries. Colour indicates the multiplier used for each pair.
    """),
    code(r"""
    one_dim = predictions["random_a_log_d1"]
    stat_y_d1 = one_dim["stat_original"][:, 0]
    stat_ay_d1 = one_dim["stat_transformed"][:, 0]

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    points = ax.scatter(
        stat_y_d1, stat_ay_d1, c=np.log10(test_a), cmap="coolwarm",
        s=15, alpha=0.50,
    )
    lower = float(min(stat_y_d1.min(), stat_ay_d1.min()))
    upper = float(max(stat_y_d1.max(), stat_ay_d1.max()))
    ax.plot([lower, upper], [lower, upper], "k--", linewidth=1.3,
            label="Perfect agreement")
    ax.set(
        xlabel=r"Original summary $s(y)$",
        ylabel=r"Scaled summary $s(ay)$",
        title="Paired agreement for the one-dimensional summary",
    )
    ax.legend()
    colourbar = fig.colorbar(points, ax=ax)
    colourbar.set_label(r"$\log_{10}(a)$")
    fig.tight_layout()
    plt.show()
    """),
    md("""
    ## 14. Two-dimensional learned summary

    The $d=2$ configuration can be visualized directly without projection. The same axes are used for the original and randomly scaled views. Similar class structure in both panels is visual evidence that the representation is stable under the random transformation while retaining model separation.
    """),
    code("""
    two_dim = predictions["random_a_log_d2"]
    stat_y = two_dim["stat_original"]
    stat_ay = two_dim["stat_transformed"]
    true_test = y_model[test_idx]

    combined = np.vstack([stat_y, stat_ay])
    x_limits = (combined[:, 0].min(), combined[:, 0].max())
    y_limits = (combined[:, 1].min(), combined[:, 1].max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, values, title in zip(
        axes, [stat_y, stat_ay], ["Original y", "Randomly scaled ay"]
    ):
        for model_id, (model_name, colour) in enumerate(zip(MODEL_NAMES, MODEL_COLOURS)):
            mask = true_test == model_id
            ax.scatter(values[mask, 0], values[mask, 1], s=13, alpha=0.35,
                       color=colour, label=model_name)
        ax.set(xlabel="Summary coordinate 1", ylabel="Summary coordinate 2", title=title)
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
    axes[1].legend(frameon=True)
    fig.suptitle("Two-dimensional learned summary before and after random scaling",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    plt.show()
    """),
    md("""
    ## 15. Model-specific classification comparison

    Normalized confusion matrices show whether increasing the dimension improves one model at the expense of another. The original view is used here so that all panels refer to the same raw held-out datasets.
    """),
    code("""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    true_test = y_model[test_idx]
    for ax, dimension in zip(axes, SUMMARY_DIMENSIONS):
        name = f"random_a_log_d{dimension}"
        predicted = predictions[name]["class_original"]
        matrix = confusion_matrix(true_test, predicted, labels=[0, 1, 2], normalize="true")
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"d={dimension}")
        ax.set_xticks(range(3), MODEL_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(3), MODEL_NAMES)
        ax.set_xlabel("Predicted model")
        if ax is axes[0]:
            ax.set_ylabel("True model")
        for row in range(3):
            for col in range(3):
                colour = "white" if matrix[row, col] > 0.5 else "#16324F"
                ax.text(col, row, f"{matrix[row, col]:.1%}",
                        ha="center", va="center", color=colour, fontsize=9)
    fig.colorbar(image, ax=axes, fraction=0.018, pad=0.02, label="Proportion within true model")
    fig.suptitle("Original-view classification as summary dimension increases",
                 fontsize=15, weight="bold")
    plt.show()
    """),
    md("""
    ## 16. Training curves for the dimension study

    These curves help distinguish a genuine dimension effect from incomplete or unstable optimization.
    """),
    code("""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for dimension in SUMMARY_DIMENSIONS:
        name = f"random_a_log_d{dimension}"
        subset = training_history[training_history["configuration"] == name]
        axes[0].plot(subset["epoch"], subset["val_loss"], linewidth=2, label=f"d={dimension}")
        axes[1].plot(subset["epoch"], subset["val_stat_difference_mse"],
                     linewidth=2, label=f"d={dimension}")
    axes[0].set(xlabel="Epoch", ylabel="Validation loss", title="Total validation objective")
    axes[1].set(xlabel="Epoch", ylabel="Validation paired MSE",
                title="Validation summary consistency")
    for ax in axes:
        ax.legend(title="Summary dimension")
    fig.suptitle("Training behaviour for random-scale log models",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    plt.show()
    """),
    md("""
    ## 17. Save held-out learned summaries

    The CSV files contain the main comparisons. The compressed NumPy file stores every held-out learned summary because dimensions 1, 2, 4, and 8 cannot be represented cleanly in one narrow table.
    """),
    code("""
    arrays_to_save = {
        "test_indices": test_idx,
        "true_model": y_model[test_idx],
        "theta": theta[test_idx],
        "random_scale_factor": scale_factor[test_idx],
    }
    for name, values in predictions.items():
        arrays_to_save[f"{name}__stat_original"] = values["stat_original"]
        arrays_to_save[f"{name}__stat_transformed"] = values["stat_transformed"]
        arrays_to_save[f"{name}__class_original"] = values["class_original"]
        arrays_to_save[f"{name}__class_transformed"] = values["class_transformed"]

    np.savez_compressed(OUTPUT_DIR / "held_out_learned_summaries.npz", **arrays_to_save)

    print("Saved outputs:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(" -", path.name)
    """),
    md(r"""
    ## 18. Interpretation guide

    The experiment should be interpreted as a three-way trade-off:

    1. **Model information:** higher held-out classification accuracy is better.
    2. **Random-scale invariance:** lower paired RMSE for $s_d(y)$ and $s_d(ay)$ is better.
    3. **Parameter leakage:** lower post-hoc probe $R^2$ is better.

    A higher-dimensional summary may improve model classification because it has more capacity. The same capacity may also preserve more information about $\theta$, so better accuracy alone does not establish a better approximately ancillary statistic.

    The scale prior broadens the previous fixed transformation to a continuum $a\in[0.1,10]$. Nevertheless, the result remains evidence of approximate invariance over this chosen prior, not a proof of exact ancillarity for every positive scale factor or every parameter change.

    Finally, the CNN remains order-sensitive even though the observations are i.i.d. A permutation-invariant encoder would be a separate architectural extension and is intentionally not introduced here, so that the current comparisons change only the requested experimental factors.
    """),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
)
nbf.write(notebook, NOTEBOOK)
print(f"Wrote {NOTEBOOK}")
