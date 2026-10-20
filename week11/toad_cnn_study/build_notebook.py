"""Build the active notebook; all results come from the revised executable study."""
from pathlib import Path
import nbformat as nbf

BASE = Path(__file__).resolve().parent
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))
def figure(name): code(f'display(Image(filename=str(FIGURES / "{name}.png")))')

md(r"""# Toad model selection: quantile-input CNN and conditional MMD

**Aim:** compress movement summaries into $d\in\{1,2,4,8\}$ learned coordinates
that distinguish three return mechanisms while reducing parameter dependence within each model.

Following the discussion with Clara, the networks use quantile summaries as inputs,
different prior-drawn parameters within the same model, and MMD between learned-summary
distributions instead of pointwise paired MSE. There is no multiplier.

The definitive training bank follows Clara's requested size exactly: 50,000 datasets
are paired with a disjoint second bank of 50,000 datasets. Pair members come from the
same model and different independently drawn parameter values.
""")
code('''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import Image, display

candidates = [Path.cwd(), Path.cwd() / "toad_cnn_study",
              Path.cwd() / "week11" / "toad_cnn_study"]
BASE = next(p for p in candidates if (p / "toad_study.py").exists())
sys.path.insert(0, str(BASE))
from toad_study import StudyConfig, run_study, build_encoder
OUTPUT = BASE / "outputs" / "quantile_mmd"
FIGURES = OUTPUT / "figures"
RUN_TRAINING = False  # True retrains; compact simulation banks are cached.
if RUN_TRAINING:
    run_study(BASE, StudyConfig())
config = json.loads((OUTPUT / "run_config.json").read_text())
display(pd.Series(config, name="Configuration"))
''')
md(r"""## 1. Models and priors

All models propose $Y^*_{t+1}=Y_t+S_t$ with symmetric stable $S_t\sim S(\alpha,\gamma)$.

| Model | Return rule | Parameters |
|---|---|---|
| Random | Return with probability $p_0$ to a previous day selected uniformly | $\alpha,\gamma,p_0$ |
| Nearest | Return with probability $p_0$ to the refuge nearest the proposal | $\alpha,\gamma,p_0$ |
| Distance-dependent | Distinct refuge weights $p_i=p_0e^{-d_i/d_0}$; no-return probability $\prod_i(1-p_i)$; conditional choice proportional to $p_i$ | $\alpha,\gamma,p_0,d_0$ |

Priors: $\alpha\sim U(1,2)$, $\gamma\sim U(10,100)$, $p_0\sim U(0,1)$,
and $d_0\sim U(20,2000)$ for the distance model. Model classes are balanced.
The study uses complete simulated histories of 66 toads over 63 days.
""")
md(r"""## 2. From movement matrices to 48 inputs

For lags $\ell=1,2,4,8$, collect $|Y_{t+\ell,j}-Y_{t,j}|$. At each lag compute:

1. the count below 10 metres;
2. the median non-return distance (at least 10 metres);
3. ten values $\log(Q_{j+1}-Q_j)$ from deciles of the **raw** non-return distances.

Thus there are $4\times12=48$ inputs, following Drovandi and Frazier (2022), section 4.4.
These are log differences of quantiles, not differences of log quantiles.
The number of non-return movements is $(63-\ell)\times66$ minus the return count.
Because the simulated histories are complete, this number is exactly determined by
the return count and is not added as a redundant 49th--52nd feature. The median and
decile gaps are computed from the actual non-return distances.
Undefined values use training-median imputation. Scaling uses training MAD, with a
training standard-deviation fallback for zero MAD. No per-dataset normalisation is used.

Toad ordering is removed by the summary construction, making **both** networks
permutation-invariant at the data-input level. The CNN now convolves over ordered
decile-gap features, not daily movement sequences. Lag identities are retained.
""")
code('''from toad_simulator import draw_parameters, simulate_toads, lag_summaries
rng = np.random.default_rng(config["seed"])
theta = draw_parameters(0, rng)
example = simulate_toads(0, theta, config["num_toads"], config["num_days"], rng)
features = lag_summaries(example)
np.testing.assert_allclose(features, lag_summaries(example[:, ::-1]), equal_nan=True)
print("Raw matrix:", example.shape, "Quantile input:", features.shape)
print("Learned dimensions:", config["summary_dimensions"])
display(pd.read_csv(OUTPUT / "preprocessing.csv").head(12))
''')
md(r"""## 3. Parameter groups, pairs and independent splits

Training contains exactly 100,000 independently simulated datasets, allocated almost
equally over the three models. Within each model, a seeded shuffle divides them into
two disjoint halves and matches them one-to-one. This produces exactly 50,000 fixed
training pairs: 16,667 random-return, 16,667 nearest-return and 16,666 distance-return.

Within the same model, pair different groups:

$$\theta_A,\theta_B\overset{\mathrm{ind}}\sim p(\theta\mid M),\qquad
Y_A\sim p(Y\mid M,\theta_A),\quad Y_B\sim p(Y\mid M,\theta_B).$$

The fixed pairing is reused across epochs and never crosses split boundaries.
All configurations share the same banks and pairs. Across MMD strengths,
each architecture/dimension starts from identical initial weights.

Validation and test parameter groups are separate. They contain repeated simulations
at fixed parameter values solely for conditional-distribution diagnostics; these extra
evaluation simulations are not part of Clara's 100,000 training datasets.

The earlier implementation paired before its validation split, contaminating validation.
Its scores are superseded. Previous outputs remain separate for provenance.
""")
code('''groups = pd.read_csv(OUTPUT / "parameter_groups.csv")
assert groups.group_id.is_unique
counts = groups.groupby(["split", "model"]).size().rename("parameter_groups").to_frame()
counts["datasets"] = counts.parameter_groups
counts.loc["train", "datasets"] *= config["train_replicates"]
counts.loc[["val", "test"], "datasets"] *= config["evaluation_replicates"]
display(counts)
print("Fixed training pairs:", config["training_pairs"])
print("Training datasets:", config["training_simulations"])
print("Total simulated datasets:", config["total_simulations"])
''')
md(r"""## 4. Classification plus conditional MMD

The encoder produces $s_\phi(Y)\in\mathbb R^d$, and only this bottleneck feeds the classifier.
MMD compares probability distributions through their kernel mean embeddings:

$$\mathrm{MMD}^2(P,Q)=
\|\mathbb E_{P}\{\varphi(S)\}-\mathbb E_Q\{\varphi(S)\}\|_{\mathcal H}^2.$$

For a characteristic kernel it is zero when the distributions are identical. Given
samples $S_A=\{s_{A,1},\ldots,s_{A,n}\}$ and
$S_B=\{s_{B,1},\ldots,s_{B,m}\}$, the biased empirical estimator is

$$\widehat{\mathrm{MMD}}^2=
\frac{1}{n^2}\sum_{i,j}k(s_{A,i},s_{A,j})
+\frac{1}{m^2}\sum_{i,j}k(s_{B,i},s_{B,j})
-\frac{2}{nm}\sum_{i,j}k(s_{A,i},s_{B,j}).$$

The first two terms measure within-sample similarity; the third measures cross-sample
similarity. The kernel averages Gaussian RBF kernels with relative bandwidth factors
$0.5,1,2$, scaled by pooled summary spread with its gradient stopped.

For each training pair,

$$L=\tfrac12[CE(M,\widehat M_A)+CE(M,\widehat M_B)]
+\lambda\widehat{\mathrm{MMD}}^2(\{s_\phi(Y_A)\},\{s_\phi(Y_B)\}),$$

where each singleton set is the empirical distribution supplied by one pair member.
For singleton samples this is a kernel distance, $2-2k(s_A,s_B)$ for a normalized
kernel. It is the distribution-distance analogue of matching paired summaries, but
it is not a high-precision estimate of the full conditional distribution.

Three Gaussian kernel bandwidths are proportional to pooled summary spread, with
gradient stopped. Uniformly shrinking a nonconstant summary does not automatically
reduce MMD. Classification and summary variance are also monitored for collapse;
these precautions do not prove the absence of every representation shortcut.

The held-out diagnostic uses repeated simulations at fixed parameter values, where
the biased estimator has a finite-sample floor. Held-out different-parameter MMD
is compared with a same-parameter split-sample control using equal sample sizes.
Dataset vectors are the samples, not coordinates of a single summary.
With 12 test replicates, six summaries are used on each side. We report

$$\mathrm{MMD}^2_{\mathrm{excess}}=
\mathrm{MMD}^2_{\mathrm{different}\,\theta}
-\mathrm{MMD}^2_{\mathrm{same}\,\theta}.$$

Low excess MMD supports approximate parameter insensitivity only if classification
remains well above chance; a collapsed constant summary would also have zero MMD.

We use $\lambda\in\{0,0.3,1\}$; $\lambda=0$ is classification-only.
""")
md(r"""## 5. CNN, flattened NN and R reference

The FFNN flattens all 48 features before dense layers.
The CNN applies shared Conv1D filters to the ten ordered decile gaps within each lag.
Return count and median are joined separately, avoiding convolution over unrelated
types of quantities. The four lag representations are then combined.

Both networks use dimensions 1, 2, 4, 8, a small classifier, Adam and validation-based
early stopping. Parameter counts differ; this is not a capacity-matched benchmark.
""")
code('''metrics = pd.read_csv(OUTPUT / "method_comparison.csv")
probes = pd.read_csv(OUTPUT / "parameter_probe_r2.csv")
history = pd.read_csv(OUTPUT / "training_history.csv")
display(metrics[["run", "trainable_parameters", "epochs", "accuracy", "accuracy_ci_low",
                 "accuracy_ci_high", "log_loss", "mean_probe_r2", "excess_mmd"]].round(4))
''')
md("""## 6. Classification and dimension

Bands are 95% bootstrap intervals resampling whole test parameter groups within each
model. Replicates share parameters, so uncertainty is not calculated as if all
replicates had independent parameter draws. Training-seed variability is not included.
""")
figure("01_accuracy_by_dimension")
figure("04_d4_regularisation_confusion_matrices")
md(r"""## 7. Parameter leakage and accuracy trade-off

After training, separate ExtraTrees probes predict each parameter from the summary
**within each model**, training on training groups and testing at unseen parameter values.
$d_0$ is evaluated only for the distance model. Parameter prediction is a diagnostic,
not a training objective.

All raw $R^2$ scores are saved. Aggregate plots average $\max(R^2,0)$ so poor scores
for one parameter do not cancel leakage in another. Low probe $R^2$ alone does not
prove independence: the probe may miss dependence, and an uninformative summary
may also give low scores.
""")
figure("02_parameter_leakage_by_dimension")
code('''baseline = metrics[metrics.mmd_weight == 0][["architecture", "dimension", "accuracy", "mean_positive_probe_r2"]]
comparison = metrics[metrics.mmd_weight > 0].merge(
    baseline, on=["architecture", "dimension"], suffixes=("", "_ce_only"))
comparison["accuracy_change"] = comparison.accuracy - comparison.accuracy_ce_only
comparison["probe_change"] = comparison.mean_positive_probe_r2 - comparison.mean_positive_probe_r2_ce_only
display(comparison[["run", "accuracy_change", "probe_change"]].round(4))
''')
md(r"""## 8. Training progress

The training curves display classification cross-entropy and accuracy at each epoch.
The parameter probe is fitted only after training, so it is not an epoch-by-epoch
quantity in this notebook.
""")
figure("10_regularised_training_progress")
md(r"""## 9. Interpretation and scaling up

Ancillarity within a model means $\mathcal L(s(Y)\mid M,\theta)$ does not depend on
$\theta$. This objective encourages distributional similarity at sampled parameter
settings; it does not establish exact ancillarity throughout the parameter space.

A useful result combines good classification, reduced parameter predictability and
little distributional change. Small training MMD by itself is insufficient.

This is a single-seed study with complete simulated observations. Repeated seeds,
more parameter groups/replicates, kernel sensitivity and the real-data missingness
pattern remain follow-up work. Neural probabilities have not been calibrated as
Bayesian posterior probabilities.

Full trajectories are discarded after summarisation, leaving compact cached banks.
There is one run configuration only: exactly 50,000 fixed training pairs formed from
100,000 unique training simulations. Pair reuse across epochs is not counted as newly
simulated data.

Revised numerical outputs are in `outputs/quantile_mmd/`. Its `figures/` folder
contains both PNG and PDF exports. `toad_simulator.py` and `toad_study.py` contain
the active neural-study implementation.  The repository-style R script is retained
only as an inactive reference file and is not used in the reported neural results.

### Sources

- Marchand et al. (2017): original movement mechanisms.
- Drovandi and Frazier (2022), *A comparison of likelihood-free methods with and without summary statistics*, section 4.4: median and log-decile-gap summaries.
- Grazian (2026), *Approximate Bayesian Computation with Statistical Distances for Model Selection*: model-selection context and priors.
- [ABC-MC toad example](https://github.com/bayesgra/ABC-MC/tree/main/toad_example): reference workflow. This study is an adaptation, not a verbatim reproduction.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.kernelspec = {"display_name": "Python 3 (.venv312)", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.12"}
nbf.write(nb, BASE / "toad_cnn_model_selection_study.ipynb")
print("Updated notebook")
