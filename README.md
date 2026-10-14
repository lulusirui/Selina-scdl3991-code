# SCDL3991 Code

This repository contains four Jupyter notebook experiments developed for an SCDL3991 research project on neural-network-based model selection and learned statistics. The corresponding Python builder scripts are included for reproducibility.

## Notebooks

### Week 7: CNN and FFNN benchmark

[`week7/cnn_vs_ffnn_benchmark.ipynb`](week7/cnn_vs_ffnn_benchmark.ipynb)

Simulates data from Exponential, Log-normal, and Gamma models and compares a one-dimensional convolutional neural network (CNN) with a feed-forward neural network (FFNN). The models are evaluated on model classification and parameter estimation.

### Week 9: DeepSets invariant statistic

[`week9/deepsets_invariant_statistic.ipynb`](week9/deepsets_invariant_statistic.ipynb)

Uses a permutation-invariant DeepSets encoder and adversarial learning to construct a low-dimensional statistic that retains model information while reducing information about the nuisance parameter.

### Week 10: Paired scale-invariant statistic

[`week10/paired_scale_invariant_statistic.ipynb`](week10/paired_scale_invariant_statistic.ipynb)

Uses paired datasets, `y` and `2y`, with a shared CNN to learn a one-dimensional statistic that remains approximately stable under a scale transformation while preserving model-classification information.

### Current experiment: random-scale invariance and summary dimension

[`week11/random_scale_summary_dimension_study.ipynb`](week11/random_scale_summary_dimension_study.ipynb)

Draws a positive random multiplier over a broad range, applies log preprocessing, and studies how learned summary dimensions 1, 2, 4, and 8 affect classification, scale invariance, and parameter leakage.

## Python builder scripts

Each experiment folder includes the Python script used to generate its notebook. These scripts provide a reproducible source version of the notebook structure and code.

## Running the notebooks

The notebooks are self-contained and generate their own simulated datasets. Open them in Jupyter Notebook, JupyterLab, VS Code, or Google Colab and run the cells in order.

The notebooks may take time to run because they train neural networks and evaluate them on simulated data.
