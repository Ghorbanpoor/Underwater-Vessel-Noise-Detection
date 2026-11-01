
import os
import re
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# Underwater Vessel Noise Challenge
# Baseline inference pipeline
#
# Input:
#   Test.csv
#   sample_submission.csv
#
# Output:
#   submission.csv
#
# This is an unsupervised, fixed per-window energy-excess
# baseline. It is NOT the trained R17 ensemble.
# ============================================================

SEED = 42

N_MINUTES = 15
N_BANDS = 25
N_FEATURES = N_MINUTES * N_BANDS

BASELINE_QUANTILE = 20.0
EXCESS_THRESHOLD_DB = 2.0
ACTIVE_THRESHOLD_DB = 3.0

FEATURE_PATTERN = re.compile(
    r"^m(?P<minute>\d+)_TOL_(?P<frequency>\d+(?:\.\d+)?)$",
    re.IGNORECASE
)


def find_feature_columns(df):
    """
    Detect and sort acoustic columns by minute and frequency.
    """
    features = []

    for col in df.columns:
        match = FEATURE_PATTERN.match(str(col).strip())

        if match:
            minute = int(match.group("minute"))
            frequency = float(match.group("frequency"))

            features.append(
                (minute, frequency, col)
            )

    features.sort(
        key=lambda item: (item[0], item[1])
    )

    if len(features) != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} acoustic columns, "
            f"but found {len(features)}."
        )

    minutes = sorted(set(x[0] for x in features))
    frequencies = sorted(set(x[1] for x in features))

    if len(minutes) != N_MINUTES:
        raise ValueError(
            f"Expected {N_MINUTES} minute positions, "
            f"found {len(minutes)}."
        )

    if len(frequencies) != N_BANDS:
        raise ValueError(
            f"Expected {N_BANDS} frequency bands, "
            f"found {len(frequencies)}."
        )

    return [x[2] for x in features]


def sigmoid(x):
    """
    Numerically stable sigmoid.
    """
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def predict_one_window(values):
    """
    Estimate the added-energy fraction using a robust
    within-window baseline for each frequency band.

    Missing cells are excluded from all energy sums.
    """

    x = np.asarray(
        values,
        dtype=np.float64
    ).reshape(N_MINUTES, N_BANDS)

    valid = np.isfinite(x)

    if not valid.any():
        return 0.0, 0.0

    # A common offset is removed for numerical stability.
    row_max = np.nanmax(x)

    # Convert dB values into relative linear energy.
    # The common scale factor cancels in the final ratio.
    energy = np.zeros_like(x, dtype=np.float64)

    energy[valid] = 10.0 ** (
        (x[valid] - row_max) / 10.0
    )

    # Estimate the background level independently
    # for each frequency band.
    baseline_db = np.full(
        N_BANDS,
        np.nan,
        dtype=np.float64
    )

    for band in range(N_BANDS):

        band_values = x[:, band]
        observed = band_values[np.isfinite(band_values)]

        if observed.size > 0:
            baseline_db[band] = np.percentile(
                observed,
                BASELINE_QUANTILE
            )

    baseline_energy = np.zeros(
        N_BANDS,
        dtype=np.float64
    )

    band_valid = np.isfinite(baseline_db)

    baseline_energy[band_valid] = 10.0 ** (
        (baseline_db[band_valid] - row_max) / 10.0
    )

    # Apply a fixed dB excess threshold.
    threshold_energy = (
        baseline_energy *
        (10.0 ** (EXCESS_THRESHOLD_DB / 10.0))
    )

    # Broadcast the frequency-specific baseline over time.
    excess = np.maximum(
        energy - threshold_energy[np.newaxis, :],
        0.0
    )

    excess[~valid] = 0.0

    total_energy = energy[valid].sum()
    added_energy = excess.sum()

    if total_energy <= 0.0:
        share = 0.0
    else:
        share = added_energy / total_energy

    share = float(
        np.clip(share, 0.0, 1.0)
    )

    # --------------------------------------------------------
    # Independent continuous evidence score.
    # This is a heuristic diagnostic score, not a calibrated
    # probability and not used in the leaderboard RMSE.
    # --------------------------------------------------------

    residual_db = np.zeros_like(x)

    for band in range(N_BANDS):
        if np.isfinite(baseline_db[band]):
            band_mask = valid[:, band]

            residual_db[band_mask, band] = (
                x[band_mask, band] - baseline_db[band]
            )

    active = (
        valid &
        (residual_db >= ACTIVE_THRESHOLD_DB)
    )

    active_fraction = (
        active.sum() / max(valid.sum(), 1)
    )

    active_band_count = np.sum(
        np.any(active, axis=0)
    )

    active_band_fraction = (
        active_band_count / N_BANDS
    )

    if active.any():
        mean_excess_db = np.mean(
            residual_db[active] - ACTIVE_THRESHOLD_DB
        )
    else:
        mean_excess_db = 0.0

    strength = sigmoid(
        (mean_excess_db - 2.0) / 2.0
    )

    temporal_evidence = np.sqrt(
        np.clip(active_fraction / 0.15, 0.0, 1.0)
    )

    spectral_evidence = np.sqrt(
        np.clip(active_band_fraction / 0.25, 0.0, 1.0)
    )

    score = (
        strength *
        temporal_evidence *
        spectral_evidence
    )

    score = float(
        np.clip(score, 0.0, 1.0)
    )

    return share, score


def run_pipeline(test_path, sample_path, output_path):

    print("=" * 60)
    print("UNDERWATER VESSEL NOISE CHALLENGE")
    print("Baseline inference pipeline")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\n[1/5] Loading Test.csv ...")

    test = pd.read_csv(test_path)

    sample = pd.read_csv(sample_path)

    if "ID" not in test.columns:
        raise ValueError("Test.csv must contain an ID column.")

    required_sample_columns = [
        "ID",
        "share_added_vessel",
        "score_added_vessel"
    ]

    for col in required_sample_columns:
        if col not in sample.columns:
            raise ValueError(
                f"Missing required column in sample submission: {col}"
            )

    if test["ID"].duplicated().any():
        raise ValueError("Duplicate IDs found in Test.csv.")

    if sample["ID"].duplicated().any():
        raise ValueError("Duplicate IDs found in sample submission.")

    if len(test) != 1500:
        print(
            f"Warning: expected 1500 test rows, "
            f"found {len(test)}."
        )

    # --------------------------------------------------------
    # Validate acoustic columns
    # --------------------------------------------------------

    print("\n[2/5] Checking acoustic features ...")

    feature_columns = find_feature_columns(test)

    print("Number of acoustic features:", len(feature_columns))

    X = test[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(dtype=np.float64)

    X = X.reshape(
        len(test),
        N_MINUTES,
        N_BANDS
    )

    missing_ratio = np.isnan(X).mean()

    print(
        f"Missing measurement ratio: "
        f"{missing_ratio:.4%}"
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print("\n[3/5] Generating predictions ...")

    predictions = []

    for i in range(len(test)):

        share, score = predict_one_window(X[i])

        predictions.append(
            (share, score)
        )

        if (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1}/{len(test)} windows"
            )

    predictions = np.asarray(
        predictions,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Create prediction dataframe
    # --------------------------------------------------------

    pred_df = pd.DataFrame({
        "ID": test["ID"].values,
        "share_added_vessel": predictions[:, 0],
        "score_added_vessel": predictions[:, 1]
    })

    # Preserve the exact ID order of sample submission.
    submission = sample[["ID"]].merge(
        pred_df,
        on="ID",
        how="left",
        validate="one_to_one"
    )

    # Fail rather than silently submit missing predictions.
    if submission[
        ["share_added_vessel", "score_added_vessel"]
    ].isna().any().any():

        raise ValueError(
            "Some sample submission IDs have no predictions. "
            "Check the IDs in Test.csv and sample submission."
        )

    # Ensure finite values and valid ranges.
    for col in [
        "share_added_vessel",
        "score_added_vessel"
    ]:

        values = submission[col].to_numpy()

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values found in {col}."
            )

        submission[col] = np.clip(
            values,
            0.0,
            1.0
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\n[4/5] Saving submission ...")

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    submission = submission[
        required_sample_columns
    ]

    submission.to_csv(
        output_path,
        index=False,
        float_format="%.10f"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n[5/5] Submission validation")

    print("Number of predictions:", len(submission))

    print(
        "Mean share prediction:",
        round(submission["share_added_vessel"].mean(), 6)
    )

    print(
        "Mean evidence score:",
        round(submission["score_added_vessel"].mean(), 6)
    )

    print(
        "Share range:",
        submission["share_added_vessel"].min(),
        "to",
        submission["share_added_vessel"].max()
    )

    print(
        "Score range:",
        submission["score_added_vessel"].min(),
        "to",
        submission["score_added_vessel"].max()
    )

    print("\nSubmission saved successfully:")
    print(output_path)

    print("\nFirst five rows:")
    print(submission.head())

    print("\nPipeline completed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Underwater Vessel Noise baseline inference"
    )

    parser.add_argument(
        "--test",
        default="data/Test.csv",
        help="Path to Test.csv"
    )

    parser.add_argument(
        "--sample",
        default="data/sample_submission.csv",
        help="Path to sample_submission.csv"
    )

    parser.add_argument(
        "--output",
        default="output/submission.csv",
        help="Output submission path"
    )

    args = parser.parse_args()

    run_pipeline(
        test_path=args.test,
        sample_path=args.sample,
        output_path=args.output
    )
