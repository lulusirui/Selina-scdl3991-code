"""Build the repaired baseline and improved Week 7 notebooks."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parent


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3.12 (SCDL Week 7)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    nbf.write(nb, ROOT / name)


common_imports = r'''
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

SEED = 10
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
print("TensorFlow:", tf.__version__)
'''

simulate = r'''
# Use 30,000 for a laptop demonstration. Increase toward 1,000,000 only after
# verifying the workflow; one million raw datasets needs substantial RAM/time.
NUM_DATASETS = 30_000
SAMPLE_SIZE = 100
N_PER_MODEL = NUM_DATASETS // 3
rng = np.random.default_rng(SEED)

# Parameters are drawn from the model-specific priors.
theta_exp = rng.exponential(scale=1.0, size=N_PER_MODEL).astype("float32")
theta_logn = rng.normal(loc=0.0, scale=1.0, size=N_PER_MODEL).astype("float32")
theta_gamma = rng.exponential(scale=1.0, size=N_PER_MODEL).astype("float32")

exp_samples = rng.exponential(
    scale=1.0 / theta_exp[:, None], size=(N_PER_MODEL, SAMPLE_SIZE)
).astype("float32")
logn_samples = rng.lognormal(
    mean=theta_logn[:, None], sigma=1.0, size=(N_PER_MODEL, SAMPLE_SIZE)
).astype("float32")
gamma_samples = rng.gamma(
    shape=2.0, scale=1.0 / theta_gamma[:, None], size=(N_PER_MODEL, SAMPLE_SIZE)
).astype("float32")

X_raw = np.vstack([exp_samples, logn_samples, gamma_samples])
y_class = np.repeat(np.arange(3), N_PER_MODEL).astype("int32")
y_param = np.concatenate([theta_exp, theta_logn, theta_gamma]).astype("float32")

# Shuffle once so all later splits are balanced without storing one-hot labels.
order = rng.permutation(len(X_raw))
X_raw, y_class, y_param = X_raw[order], y_class[order], y_param[order]
print(X_raw.shape, y_class.shape, y_param.shape)
print("Approximate raw-data memory (MB):", round(X_raw.nbytes / 1024**2, 1))
'''

test_data = r'''
N_TEST_PER_MODEL = 100
test_rng = np.random.default_rng(42)
test_exp = test_rng.exponential(scale=2.0, size=(N_TEST_PER_MODEL, SAMPLE_SIZE))
test_logn = test_rng.lognormal(
    mean=np.log(2.0) - 0.5, sigma=1.0, size=(N_TEST_PER_MODEL, SAMPLE_SIZE)
)
test_gamma = test_rng.gamma(shape=2.0, scale=1.0, size=(N_TEST_PER_MODEL, SAMPLE_SIZE))
X_new_raw = np.vstack([test_exp, test_logn, test_gamma]).astype("float32")
y_new = np.repeat(np.arange(3), N_TEST_PER_MODEL)
theta_new = np.concatenate([
    np.full(N_TEST_PER_MODEL, 0.5),
    np.full(N_TEST_PER_MODEL, np.log(2.0) - 0.5),
    np.full(N_TEST_PER_MODEL, 1.0),
]).astype("float32")
'''

diagnostics = r'''
pred_prob, pred_param = model.predict(X_new, verbose=0)
pred_class = pred_prob.argmax(axis=1)  # deterministic MAP decision
cm = confusion_matrix(y_new, pred_class)
accuracy = np.mean(pred_class == y_new)
mae = np.mean(np.abs(pred_param[:, 0] - theta_new))
print(f"Fixed-parameter test accuracy: {accuracy:.3f}")
print(f"Fixed-parameter test MAE: {mae:.3f}")
print(cm)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(cm, display_labels=MODEL_NAMES).plot(ax=axes[0], colorbar=False)
axes[0].set_title("Model classification")
axes[1].scatter(theta_new, pred_param[:, 0], c=y_new, cmap="viridis", alpha=0.65)
lo = min(theta_new.min(), pred_param.min())
hi = max(theta_new.max(), pred_param.max())
axes[1].plot([lo, hi], [lo, hi], "k--", linewidth=1)
axes[1].set(xlabel="True parameter", ylabel="NN estimate", title="Parameter estimates")
plt.tight_layout()
plt.show()

pd.DataFrame(pred_prob, columns=MODEL_NAMES).to_csv(
    OUTPUT_DIR / f"{OUTPUT_STEM}_probabilities.csv", index=False
)
pd.DataFrame({"true_model": y_new, "true_theta": theta_new,
              "nn_theta": pred_param[:, 0]}).to_csv(
    OUTPUT_DIR / f"{OUTPUT_STEM}_parameters.csv", index=False
)
'''


baseline = [
    md(r'''
# Repaired baseline neural network with prior simulation

This notebook stays close to Clara's simple feed-forward multi-task network. It fixes the environment and two important data issues in the original:

1. the regression target is now the parameter actually drawn from each prior, rather than a fixed test value;
2. the default reference table is laptop-sized (30,000 datasets), while retaining a single setting that can be increased for a larger run.

The input is the `arcsinh`-transformed raw sample. The two outputs estimate the model probability and model-specific parameter.
'''),
    code(common_imports + '\nMODEL_NAMES = ["Exponential", "Log-normal", "Gamma"]\nOUTPUT_DIR = Path("week7") if Path("week7").is_dir() else Path.cwd()\nOUTPUT_STEM = "expo_baseline"'),
    code(simulate),
    code(r'''
X = np.arcsinh(X_raw).astype("float32")
X_train, X_test, yc_train, yc_test, yp_train, yp_test = train_test_split(
    X, y_class, y_param, test_size=0.20, random_state=42, stratify=y_class
)
normalizer = layers.Normalization()
normalizer.adapt(X_train)
'''),
    code(r'''
inputs = keras.Input(shape=(SAMPLE_SIZE,), name="sample")
x = normalizer(inputs)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(64, activation="relu", name="embedding")(x)
class_output = layers.Dense(3, activation="softmax", name="class_output")(x)
param_output = layers.Dense(1, name="param_output")(x)
model = keras.Model(inputs, [class_output, param_output], name="baseline_multitask_nn")
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss={"class_output": "sparse_categorical_crossentropy", "param_output": "mse"},
    loss_weights={"class_output": 1.0, "param_output": 0.5},
    metrics={"class_output": ["accuracy"], "param_output": ["mae"]},
)
model.summary()
'''),
    code(r'''
callbacks = [keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)]
history = model.fit(
    X_train, {"class_output": yc_train, "param_output": yp_train},
    validation_split=0.20, epochs=12, batch_size=256,
    callbacks=callbacks, verbose=2,
)
scores = model.evaluate(
    X_test, {"class_output": yc_test, "param_output": yp_test}, verbose=0,
    return_dict=True,
)
scores
'''),
    code(test_data + '\nX_new = np.arcsinh(X_new_raw).astype("float32")'),
    code(diagnostics),
    md(r'''
## Interpretation

The softmax values approximate model posterior probabilities under the training prior only to the extent that the classifier is calibrated and the simulations match the assumed generative mechanism. For formal ABC model choice, the learned embedding can instead be used as a low-dimensional summary and followed by an ABC accept/reject or weighted correction step.
'''),
]


improved = [
    md(r'''
# Improved permutation-invariant summary network

The observations in each dataset are i.i.d., so their row order should not affect the answer. A dense network on 100 raw positions does not enforce that property. This version first maps each dataset to a compact set of order-invariant empirical summaries, then uses a regularised residual multi-task network.

Improvements over the baseline:

- permutation-invariant summaries appropriate for i.i.d. samples;
- true prior-drawn parameter labels;
- standardisation learned from the training split only;
- batch normalisation, dropout, an L2 penalty, early stopping, and learning-rate reduction;
- deterministic MAP classification and explicit calibration/parameter diagnostics.
'''),
    code(common_imports + '\nMODEL_NAMES = ["Exponential", "Log-normal", "Gamma"]\nOUTPUT_DIR = Path("week7") if Path("week7").is_dir() else Path.cwd()\nOUTPUT_STEM = "expo_summary_multitask"'),
    code(simulate),
    code(r'''
def summary_features(x):
    """Permutation-invariant features, including the known sufficient-statistic terms."""
    x = np.asarray(x, dtype="float32")
    safe = np.clip(x, 1e-7, None)
    logx = np.log(safe)
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    centered = x - mean[:, None]
    skew = (centered**3).mean(axis=1) / np.maximum(std, 1e-6)**3
    q = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T
    return np.column_stack([
        np.arcsinh(mean), np.arcsinh(std), np.arcsinh(skew),
        logx.mean(axis=1), (logx**2).mean(axis=1),
        np.arcsinh(q),
    ]).astype("float32")

X = summary_features(X_raw)
FEATURE_NAMES = ["asinh_mean", "asinh_sd", "asinh_skew", "mean_log_x",
                 "mean_log_x2", "q10", "q25", "q50", "q75", "q90"]
print("Summary matrix:", X.shape)
pd.DataFrame(X[:5], columns=FEATURE_NAMES)
'''),
    code(r'''
X_train, X_test, yc_train, yc_test, yp_train, yp_test = train_test_split(
    X, y_class, y_param, test_size=0.20, random_state=42, stratify=y_class
)
normalizer = layers.Normalization()
normalizer.adapt(X_train)

inputs = keras.Input(shape=(X.shape[1],), name="summaries")
x0 = normalizer(inputs)
x = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-4))(x0)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(0.15)(x)
x = layers.Dense(64, activation="relu")(x)
skip = layers.Dense(64)(x0)
embedding = layers.Add(name="embedding")([x, skip])
embedding = layers.Activation("relu")(embedding)
class_output = layers.Dense(3, activation="softmax", name="class_output")(embedding)
param_output = layers.Dense(1, name="param_output")(embedding)

model = keras.Model(inputs, [class_output, param_output], name="summary_residual_multitask_nn")
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss={"class_output": "sparse_categorical_crossentropy", "param_output": "huber"},
    loss_weights={"class_output": 1.0, "param_output": 0.5},
    metrics={"class_output": ["accuracy"], "param_output": ["mae"]},
)
model.summary()
'''),
    code(r'''
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
]
history = model.fit(
    X_train, {"class_output": yc_train, "param_output": yp_train},
    validation_split=0.20, epochs=20, batch_size=256,
    callbacks=callbacks, verbose=2,
)
scores = model.evaluate(
    X_test, {"class_output": yc_test, "param_output": yp_test}, verbose=0,
    return_dict=True,
)
scores
'''),
    code(test_data + '\nX_new = summary_features(X_new_raw)'),
    code(diagnostics),
    code(r'''
# Sanity check: permuting observations within each dataset must not change features/predictions.
permuted = np.stack([test_rng.permutation(row) for row in X_new_raw])
max_feature_difference = np.max(np.abs(summary_features(permuted) - X_new))
print("Maximum feature difference after within-row permutation:", max_feature_difference)
assert max_feature_difference < 1e-5
'''),
    md(r'''
## How this connects to NN-ABC

The 64-dimensional `embedding` is a learned summary. A next experiment can simulate a large ABC reference table, compute this embedding for observed and simulated datasets, retain the closest fraction, and compare the resulting NN-ABC posterior with the standalone softmax output. This implements the report's idea that ABC can correct a biased neural estimate, although the amount of correction and the posterior spread remain problem-dependent.
'''),
]


save("cnn_vs_ffnn_benchmark.ipynb", baseline)
save("expo_example_summary_multitask.ipynb", improved)

# Keep the fixed notebook as a deliberately minimal patch of Clara's original.
# The more substantial architectural experiment remains in the separate
# expo_example_summary_multitask.ipynb notebook.
original_path = ROOT / "expo_example_NNwithprior.ipynb"
minimal = nbf.read(original_path, as_version=4)
minimal["metadata"]["kernelspec"] = {
    "display_name": "Python 3.12 (SCDL Week 7)",
    "language": "python",
    "name": "python3",
}
minimal["metadata"].setdefault("language_info", {})["version"] = "3.12"
minimal["metadata"]["week7_minimal_fix"] = {
    "source": "expo_example_NNwithprior.ipynb",
    "changes": [
        "Use Python 3.12 TensorFlow environment",
        "Train parameter output on the actual prior-drawn theta values",
        "Correct an inaccurate probability-shape comment from 6 to 3 classes",
    ],
}
minimal["cells"][7]["source"] = r'''# Creation of parameter labels

# FIX: use the parameters actually drawn from the priors in Cell 3.
# The original cell used the three fixed test values for every training row,
# so the regression head could not learn the simulated parameters.
y_params = np.concatenate([theta_exp, theta_logn, theta_gamma]).reshape(-1, 1)

y_params.shape
'''
minimal["cells"][15]["source"] = "".join(minimal["cells"][15]["source"]).replace(
    "shape (n_samples, 6)", "shape (n_samples, 3)"
).replace("all 6 model probabilities", "all 3 model probabilities")
for cell in minimal["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
nbf.write(minimal, ROOT / "cnn_vs_ffnn_benchmark.ipynb")
print("Built both notebooks in", ROOT)
