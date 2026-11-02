
"""Underwater Vessel Noise Challenge - evidence-based conservative baseline.

The supplied project files contain unlabelled South African background
measurements and descriptive EDA, but no labelled controlled-addition targets.
Consequently this script does NOT pretend to train a supervised share model.

Default leaderboard strategy:
    share_added_vessel = 0.0

The supplied deep-EDA summary records an all-zero reference score of 0.753867825.
This is a reference from that analysis snapshot, not a guarantee of future score.

score_added_vessel is a separate, heuristic acoustic-evidence diagnostic. The
challenge documentation says this column does not affect the automatic score.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

N_MINUTES = 15
N_BANDS = 25
N_FEATURES = N_MINUTES * N_BANDS

FEATURE_RE = re.compile(
    r"^m(?P<minute>\d+)_TOL_(?P<frequency>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return acoustic columns ordered by minute, then band frequency."""
    found = []

    for col in frame.columns:
        match = FEATURE_RE.match(str(col).strip())

        if match:
            found.append((
                int(match.group("minute")),
                float(match.group("frequency")),
                str(col)
            ))

    found.sort(key=lambda item: (item[0], item[1]))

    if len(found) != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} acoustic columns; found {len(found)}."
        )

    minutes = sorted({item[0] for item in found})
    frequencies = sorted({item[1] for item in found})

    if len(minutes) != N_MINUTES or len(frequencies) != N_BANDS:
        raise ValueError(
            f"Unexpected layout: {len(minutes)} minutes and "
            f"{len(frequencies)} frequency bands."
        )

    return [item[2] for item in found]


def sigmoid(value: float) -> float:
    value = float(np.clip(value, -30.0, 30.0))
    return 1.0 / (1.0 + np.exp(-value))


def safe_nanmedian(values: np.ndarray, axis: int) -> np.ndarray:
    """Nanmedian without noisy all-NaN runtime warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def acoustic_evidence_score(
    window: np.ndarray,
    frequencies: np.ndarray
) -> tuple[float, dict]:
    """Calculate a heuristic evidence score; not a calibrated probability."""

    x = np.asarray(window, dtype=np.float64).reshape(
        N_MINUTES, N_BANDS
    )

    valid = np.isfinite(x)

    if not valid.any():
        return 0.0, {
            "observed_minutes": 0,
            "low_peak_residual_db": 0.0,
            "low_active_fraction": 0.0,
            "active_band_fraction": 0.0,
        }

    # Robust reference level for each frequency band.
    band_reference = safe_nanmedian(
        np.where(valid, x, np.nan),
        axis=0
    )

    residual = x - band_reference[np.newaxis, :]
    residual[~valid] = np.nan

    # Low-frequency bands up to 500 Hz.
    low_bands = frequencies <= 500.0

    if not low_bands.any():
        raise ValueError(
            "No frequency bands at or below 500 Hz were found."
        )

    low_residual = residual[:, low_bands]

    minute_low_level = safe_nanmedian(
        low_residual,
        axis=1
    )

    minute_low_level = minute_low_level[
        np.isfinite(minute_low_level)
    ]

    if minute_low_level.size:
        low_peak = float(np.nanmax(minute_low_level))
    else:
        low_peak = 0.0

    low_valid = valid[:, low_bands]

    low_active = low_valid & (
        np.nan_to_num(low_residual, nan=-1e6) >= 5.0
    )

    low_active_fraction = float(
        low_active.sum() / max(int(low_valid.sum()), 1)
    )

    band_active = np.any(
        valid & (
            np.nan_to_num(residual, nan=-1e6) >= 5.0
        ),
        axis=0
    )

    active_band_fraction = float(band_active.mean())

    # Broad-band temporal change.
    all_minute_residual = safe_nanmedian(
        residual,
        axis=1
    )

    all_minute_residual = all_minute_residual[
        np.isfinite(all_minute_residual)
    ]

    if all_minute_residual.size >= 2:
        temporal_step = float(
            np.nanmedian(
                np.abs(np.diff(all_minute_residual))
            )
        )
    else:
        temporal_step = 0.0

    # Acoustic evidence components.
    strength = sigmoid(
        (low_peak - 4.0) / 2.5
    )

    temporal_support = np.sqrt(
        np.clip(low_active_fraction / 0.20, 0.0, 1.0)
    )

    spectral_support = np.sqrt(
        np.clip(active_band_fraction / 0.25, 0.0, 1.0)
    )

    dynamics_support = (
        0.75
        + 0.25 * sigmoid(
            (temporal_step - 1.5) / 1.5
        )
    )

    score = float(np.clip(
        strength
        * temporal_support
        * spectral_support
        * dynamics_support,
        0.0,
        1.0
    ))

    diagnostics = {
        "observed_minutes": int(valid.any(axis=1).sum()),
        "low_peak_residual_db": low_peak,
        "low_active_fraction": low_active_fraction,
        "active_band_fraction": active_band_fraction,
        "median_broadband_step_db": temporal_step,
    }

    return score, diagnostics


def build_submission(
    test_path: Path,
    sample_path: Path,
    output_path: Path,
    diagnostics_path: Path
) -> pd.DataFrame:

    print("=" * 64)
    print("UNDERWATER VESSEL NOISE CHALLENGE")
    print("Conservative share baseline + acoustic evidence diagnostic")
    print("=" * 64)

    test = pd.read_csv(
        test_path,
        dtype={"ID": str}
    )

    sample = pd.read_csv(
        sample_path,
        dtype={"ID": str}
    )

    required = [
        "ID",
        "share_added_vessel",
        "score_added_vessel"
    ]

    if "ID" not in test.columns:
        raise ValueError(
            "Test.csv must contain an ID column."
        )

    if sample.columns.tolist() != required:
        raise ValueError(
            f"Sample submission columns must be exactly: {required}"
        )

    if test["ID"].isna().any() or sample["ID"].isna().any():
        raise ValueError("Missing IDs were found.")

    if test["ID"].duplicated().any():
        raise ValueError("Duplicate IDs were found in Test.csv.")

    if sample["ID"].duplicated().any():
        raise ValueError(
            "Duplicate IDs were found in sample submission."
        )

    if set(test["ID"]) != set(sample["ID"]):
        raise ValueError(
            "Test IDs and sample-submission IDs do not match."
        )

    feature_columns = get_feature_columns(test)

    frequencies = np.array([
        float(
            FEATURE_RE.match(c).group("frequency")
        )
        for c in feature_columns
    ], dtype=np.float64).reshape(
        N_MINUTES, N_BANDS
    )[0]

    values = test[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    data = values.to_numpy(
        dtype=np.float64
    ).reshape(
        -1, N_MINUTES, N_BANDS
    )

    print(f"Rows: {len(test):,}")
    print(f"Acoustic shape: {data.shape}")
    print(
        f"Missing-cell ratio: {np.isnan(data).mean():.3%}"
    )

    print("Generating acoustic evidence scores ...")

    scores = np.zeros(
        len(test),
        dtype=np.float64
    )

    diagnostic_rows = []

    for i, window in enumerate(data):

        scores[i], diagnostic = acoustic_evidence_score(
            window,
            frequencies
        )

        diagnostic_rows.append({
            "ID": test.iloc[i]["ID"],
            **diagnostic
        })

        if (i + 1) % 250 == 0 or i + 1 == len(test):
            print(
                f"Processed {i + 1:,}/{len(test):,}"
            )

    # No labelled controlled-addition targets are included
    # in the supplied archive. Therefore, use the all-zero
    # reference baseline instead of inventing model predictions.

    predictions = pd.DataFrame({
        "ID": test["ID"].to_numpy(),
        "share_added_vessel": np.zeros(
            len(test),
            dtype=np.float64
        ),
        "score_added_vessel": scores,
    })

    # Preserve the exact sample-submission ID order.
    submission = sample[["ID"]].merge(
        predictions,
        on="ID",
        how="left",
        validate="one_to_one"
    )

    if submission[required[1:]].isna().any().any():
        raise ValueError(
            "One or more IDs did not receive predictions."
        )

    for column in required[1:]:

        arr = submission[column].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(arr).all():
            raise ValueError(
                f"Non-finite predictions in {column}."
            )

        submission[column] = np.clip(
            arr,
            0.0,
            1.0
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    diagnostics_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    submission[required].to_csv(
        output_path,
        index=False,
        float_format="%.10f"
    )

    pd.DataFrame(diagnostic_rows).to_csv(
        diagnostics_path,
        index=False
    )

    print("\nSubmission validation")
    print(f"Rows: {len(submission):,}")

    print(
        f"Mean share: "
        f"{submission['share_added_vessel'].mean():.6f}"
    )

    print(
        f"Mean evidence score: "
        f"{submission['score_added_vessel'].mean():.6f}"
    )

    print(f"Saved submission: {output_path}")
    print(f"Saved diagnostics: {diagnostics_path}")

    print(
        "NOTE: score_added_vessel is a heuristic diagnostic, "
        "not a calibrated probability."
    )

    return submission


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create a conservative underwater "
            "vessel-noise submission."
        )
    )

    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/Test.csv")
    )

    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("data/sample_submission.csv")
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/submission.csv")
    )

    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=Path(
            "output/prediction_diagnostics.csv"
        )
    )

    args = parser.parse_args()

    build_submission(
        args.test,
        args.sample,
        args.output,
        args.diagnostics
    )


if __name__ == "__main__":
    main()
