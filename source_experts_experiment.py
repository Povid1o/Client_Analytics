"""Observable source-availability experts with a soft OOF blend."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_COLS,
    CSV_READ_KWARGS,
    DATE_COL,
    ID_COL,
    NON_FEATURE_COLS,
    PARTIAL_OUTPUTS_DIR,
    RANDOM_SEED,
    TARGET_COL,
    TRAIN_PATH,
    WEIGHT_COL,
)
from src.feature_engineering import INCOME_ANCHORS
from src.metrics import wmae
from src.modeling import PowerTargetCatBoost, prepare_cat_features
from src.postprocessing import blend_salary_signal
from src.preprocessing import preprocess


REFERENCE_ALPHAS = {
    "salary_present": 0.85,
    "first_salary_present": 1.00,
    "multi_anchor_ge3": 0.85,
    "sparse_anchor_le1": 0.45,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-predictions",
        default=PARTIAL_OUTPUTS_DIR / "residual_stacking_outer_predictions.csv",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--output",
        default=PARTIAL_OUTPUTS_DIR / "source_experts_results.csv",
    )
    return parser.parse_args()


def build_matrix(frame):
    features = [column for column in frame.columns if column not in NON_FEATURE_COLS]
    matrix = frame[features].copy()
    matrix[DATE_COL] = matrix[DATE_COL].astype(str)
    return prepare_cat_features(matrix, CATEGORICAL_COLS)


def segment_masks(frame):
    anchor_count = frame[INCOME_ANCHORS].notna().sum(axis=1).to_numpy()
    return {
        "salary_present": frame["salary_6to12m_avg"].notna().to_numpy(),
        "first_salary_present": frame["first_salary_income"].notna().to_numpy(),
        "multi_anchor_ge3": anchor_count >= 3,
        "sparse_anchor_le1": anchor_count <= 1,
    }


def main():
    args = parse_args()
    started = time.time()
    raw = pd.read_csv(TRAIN_PATH, **CSV_READ_KWARGS, low_memory=False)
    frame, _ = preprocess(raw, is_train=True)
    outer = pd.read_csv(args.base_predictions)
    validation_ids = set(outer[ID_COL])
    train_mask = ~frame[ID_COL].isin(validation_ids).to_numpy()
    val_mask = ~train_mask
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    y = frame[TARGET_COL].to_numpy(dtype=float)
    weights = frame[WEIGHT_COL].to_numpy(dtype=float)
    X = build_matrix(frame)
    base = outer.set_index(ID_COL).loc[frame.iloc[val_idx][ID_COL], "base_prediction"].to_numpy()
    base_score = wmae(y[val_idx], base, weights[val_idx])
    masks = segment_masks(frame)

    results = []
    prediction_output = {
        ID_COL: frame.iloc[val_idx][ID_COL].to_numpy(),
        TARGET_COL: y[val_idx],
        WEIGHT_COL: weights[val_idx],
        "base_prediction": base,
    }
    for segment_i, (name, mask) in enumerate(masks.items()):
        selected_train = train_mask & mask
        selected_val = mask[val_idx]
        model = PowerTargetCatBoost(
            target_power=0.25,
            cat_features=CATEGORICAL_COLS,
            params={
                "iterations": args.iterations,
                "learning_rate": 0.05,
                "depth": 7,
                "l2_leaf_reg": 15,
                "random_seed": RANDOM_SEED + 501 + segment_i,
                "early_stopping_rounds": None,
            },
        )
        local_train_idx = np.flatnonzero(selected_train)
        model.fit(X.iloc[local_train_idx], y[local_train_idx], weights[local_train_idx])
        expert = model.predict(X.iloc[val_idx])
        expert = blend_salary_signal(
            expert,
            frame.iloc[val_idx]["salary_6to12m_avg"].to_numpy(),
            alpha=0.6,
        )
        expert = np.clip(expert, y.min(), y.max())
        segment_base_score = wmae(
            y[val_idx][selected_val], base[selected_val], weights[val_idx][selected_val]
        )
        segment_expert_score = wmae(
            y[val_idx][selected_val], expert[selected_val], weights[val_idx][selected_val]
        )
        best = (base_score, 0.0)
        for alpha in np.arange(0, 1.01, 0.05):
            prediction = base.copy()
            prediction[selected_val] = (
                (1 - alpha) * base[selected_val] + alpha * expert[selected_val]
            )
            score = wmae(y[val_idx], prediction, weights[val_idx])
            if score < best[0]:
                best = (score, alpha)
        fixed_alpha = REFERENCE_ALPHAS[name]
        fixed_prediction = base.copy()
        fixed_prediction[selected_val] = (
            (1 - fixed_alpha) * base[selected_val]
            + fixed_alpha * expert[selected_val]
        )
        fixed_score = wmae(y[val_idx], fixed_prediction, weights[val_idx])
        results.append(
            {
                "segment": name,
                "train_rows": int(selected_train.sum()),
                "val_rows": int(selected_val.sum()),
                "segment_base_wmae": segment_base_score,
                "segment_expert_wmae": segment_expert_score,
                "best_alpha": best[1],
                "blend_wmae": best[0],
                "delta": best[0] - base_score,
                "reference_alpha": fixed_alpha,
                "reference_blend_wmae": fixed_score,
                "reference_delta": fixed_score - base_score,
            }
        )
        prediction_output[f"expert_{name}"] = expert
        prediction_output[f"gate_{name}"] = selected_val.astype("int8")
        print(
            f"{name:<22} train={selected_train.sum():>6,} val={selected_val.sum():>5,} "
            f"local base/expert={segment_base_score:,.0f}/{segment_expert_score:,.0f} "
            f"blend={best[0]:,.0f} delta={best[0]-base_score:+,.0f} alpha={best[1]:.2f}",
            flush=True,
        )

    result = pd.DataFrame(results).sort_values("blend_wmae")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    prediction_path = output_path.with_name(f"{output_path.stem}_predictions.csv")
    pd.DataFrame(prediction_output).to_csv(prediction_path, index=False)
    print(f"base WMAE={base_score:,.0f}")
    print(result.to_string(index=False))
    print(f"saved {args.output} and {prediction_path}; time={time.time()-started:.1f}s")


if __name__ == "__main__":
    main()
