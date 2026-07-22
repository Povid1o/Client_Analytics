"""Three-model conditional lognormal head for exact WMAE postprocessing."""

import lightgbm as lgb
import numpy as np

from src.wmae_distribution import log_quantiles_to_weighted_lognormal


QUANTILE_ROUNDS = {0.20: 1845, 0.50: 3000, 0.80: 2382}
QUANTILE_SEEDS = {0.20: 12501, 0.50: 12502, 0.80: 12503}


def quantile_params(quantile):
    return {
        "objective": "quantile",
        "alpha": quantile,
        "n_estimators": QUANTILE_ROUNDS[quantile],
        "learning_rate": 0.012,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 120,
        "colsample_bytree": 0.82,
        "subsample": 0.90,
        "subsample_freq": 1,
        "reg_alpha": 5.0,
        "reg_lambda": 45.0,
        "random_state": QUANTILE_SEEDS[quantile],
        "n_jobs": -1,
        "verbosity": -1,
    }


def fit_predict_weighted_lognormal(train_features, target, inference_features, rule):
    """Fit q20/q50/q80 on log(target), then apply exact WMAE weighting."""
    log_target = np.log(np.maximum(np.asarray(target, dtype=float), 1e-12))
    predictions = []
    for quantile in QUANTILE_ROUNDS:
        model = lgb.LGBMRegressor(**quantile_params(quantile))
        model.fit(train_features, log_target)
        predictions.append(model.predict(inference_features))
        print(
            f"trained weighted-lognormal quantile q={quantile:.2f} "
            f"({len(predictions)}/3)", flush=True,
        )
    predictions = np.maximum.accumulate(np.column_stack(predictions), axis=1)
    weighted = log_quantiles_to_weighted_lognormal(
        predictions[:, 0], predictions[:, 1], predictions[:, 2], rule
    )
    return weighted, predictions
