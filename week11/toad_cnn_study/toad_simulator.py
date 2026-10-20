"""Paper-defined toad simulator and 48 lag summaries (no neural-network code)."""
import numpy as np
from scipy.stats import levy_stable

MODEL_NAMES = ["Random return", "Nearest return", "Distance-dependent return"]
PARAMETER_NAMES = ["alpha", "gamma", "p0", "d0"]
LAGS = (1, 2, 4, 8)
SUMMARY_NAMES = [
    f"lag{lag}_{feature}"
    for lag in LAGS
    for feature in ["return_count", "median", *[f"log_decile_gap_{i}" for i in range(10)]]
]


def draw_parameters(model_id, rng):
    return np.array([rng.uniform(1, 2), rng.uniform(10, 100), rng.uniform(0, 1),
                     rng.uniform(20, 2000) if model_id == 2 else np.nan])


def simulate_toads(model_id, theta, num_toads, num_days, rng):
    """Locations: [day, toad]; stable steps and paper-defined refuge rules."""
    alpha, gamma, p0, d0 = theta
    positions = np.zeros((num_days, num_toads), dtype=np.float64)
    steps = levy_stable.rvs(alpha, 0, scale=gamma,
                           size=(num_days - 1, num_toads), random_state=rng)
    refuges = np.zeros_like(positions)
    counts = np.ones(num_toads, dtype=int)
    for day in range(1, num_days):
        proposed = positions[day - 1] + steps[day - 1]
        positions[day] = proposed
        if model_id in (0, 1):
            returning = np.flatnonzero(rng.random(num_toads) < p0)
            if returning.size:
                if model_id == 0:
                    indices = rng.integers(0, day, size=returning.size)
                else:
                    indices = np.argmin(np.abs(positions[:day, returning] - proposed[returning]), axis=0)
                positions[day, returning] = positions[indices, returning]
        elif model_id == 2:
            n_refuges = int(counts.max())
            active = np.arange(n_refuges)[:, None] < counts[None, :]
            probabilities = p0 * np.exp(-np.abs(proposed - refuges[:n_refuges]) / d0)
            probabilities *= active
            no_return = np.prod(1 - probabilities, axis=0)
            returning = np.flatnonzero(rng.random(num_toads) >= no_return)
            if returning.size:
                weights = probabilities[:, returning]
                cumulative = np.cumsum(weights, axis=0)
                draws = rng.random(len(returning)) * cumulative[-1]
                indices = np.sum(cumulative <= draws, axis=0)
                positions[day, returning] = refuges[indices, returning]
            staying = np.setdiff1d(np.arange(num_toads), returning)
            # Continuous proposals almost surely create distinct new refuges.
            refuges[counts[staying], staying] = proposed[staying]
            counts[staying] += 1
        else:
            raise ValueError(model_id)
    return positions


def lag_summaries(positions, threshold=10.0):
    """Each lag: count, raw median, log of ten adjacent RAW decile gaps.

    Drovandi & Frazier (2022), section 4.4: log(Q[j+1]-Q[j]), not
    log(Q[j+1])-log(Q[j]). Undefined values become NaN, then are imputed
    by TRAINING medians. No per-dataset standardisation is performed.
    """
    result = []
    for lag in LAGS:
        distances = np.abs(positions[lag:] - positions[:-lag]).ravel()
        distances = distances[np.isfinite(distances)]
        nonreturn = distances[distances >= threshold]
        count = np.count_nonzero(distances < threshold)
        if nonreturn.size:
            quantiles = np.quantile(nonreturn, np.linspace(0, 1, 11))
            gaps = np.diff(quantiles)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_gaps = np.where(gaps > 0, np.log(gaps), np.nan)
            result.extend([count, np.median(nonreturn), *log_gaps])
        else:
            result.extend([count, *([np.nan] * 11)])
    return np.asarray(result, dtype=np.float32).reshape(4, 12)
