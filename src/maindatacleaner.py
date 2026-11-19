import json
import os
import numpy as np
import pandas as pd

CONFIG = {
    "data_path": "data/ProjectData/proj_machine_readings.csv",
    "cleaned_output_path": "data/ProjectData/proj_machine_readings_cleaned.csv",
    "target_column": "maintenance_required",
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",  # Helpful if maintenance_required=1 is rarer
    "use_feature_engineering": True,
    "save_model_dir": "saved_model",
    "output_plot_path": "data/feature_importances.png",
}

# Change made: Set global_bounds that apply to all datasets(for now)
global_bounds = {
        "temperature": (None, None),
        "vibration": (0.0, None),
        "humidity": (0.0, 100.0),
        "pressure": (0.0, None),
        "energy": (0.0, None),
        "consumption": (0.0, None),
        "predicted_remaining_life": (0.0, None),
        "downtime_risk": (0.0, 1.0),
}

# ran into a small issue of some columns being dictionaries,
# had to change them to make them hashable for data ingestion
def _make_hashable(val):
    if isinstance(val, (dict, list)):
        return json.dumps(val, sort_keys=True) if isinstance(val, dict) else json.dumps(val)
    return val

def ingest_data(csv_path: str) -> pd.DataFrame:
    """Ingest raw CSV data from disk."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at '{csv_path}'.")
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame, target_column: str = "maintenance_required") -> pd.DataFrame:
    """Clean duplicates, types, invalid ranges, and missing values."""

    # required line to make certain data values hashable
    for col in df.columns:
        if df[col].dtype == object and df[col].map(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].map(_make_hashable)

    df = df.copy().drop_duplicates()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").drop(columns=["timestamp"])

    if "machine_id" in df.columns:
        df["machine_id"] = df["machine_id"].astype(str).str.strip()

    if "failure_type" in df.columns:
        df["failure_type"] = df["failure_type"].astype(str).str.strip()
        df.loc[df["failure_type"].isin(["", "nan", "None"]), "failure_type"] = "Unknown"

    # additional input here to fix some sign-flipped negative vibration sensor glitching
    if "vibration" in df.columns:
        df["vibration"] = pd.to_numeric(df["vibration"], errors="coerce").abs()

    for col, (low, high) in global_bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if low is not None or high is not None:
                df[col] = df[col].clip(lower=low, upper=high)

    for col in ["machine_status", "anomaly_flag", "maintenance_required"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # extra cleaning check to remove duplicated column names (when necessary)
    if isinstance(target_column, dict):
        target_column = target_column.get("name") or target_column.get("column")

    if isinstance(target_column, str) and target_column in df.columns:
        # Deduplicate column names if the CSV has repeated headers
        if df.columns.tolist().count(target_column) > 1:
            first_idx = df.columns.tolist().index(target_column)
            target_series = df.iloc[:, first_idx]
            df = df.loc[:, ~df.columns.duplicated()].copy()
            df[target_column] = target_series

        df = df.dropna(subset=[target_column])
        df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
        df = df.dropna(subset=[target_column])
        df[target_column] = df[target_column].astype(int)

    num_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in {"machine_id", target_column}
    ]
    for col in num_cols:
        if "machine_id" in df.columns:
            machine_medians = df.groupby("machine_id")[col].transform("median")
            df[col] = df[col].fillna(machine_medians)
        # fix machine to fallback on global_median if entire columns are NaN or missing
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df

def engineer_features(df: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
    """Create domain features from sensor and derived columns."""
    df = df.copy()
    if not enabled:
        return df

    eps = 1e-5
    if {"temperature", "pressure"}.issubset(df.columns):
        df["temp_pressure_ratio"] = df["temperature"] / (df["pressure"].abs() + eps)

    if {"energy", "consumption"}.issubset(df.columns):
        df["energy_per_consumption"] = df["energy"] / (df["consumption"].abs() + eps)

    if {"vibration", "temperature"}.issubset(df.columns):
        df["thermal_vibration_index"] = df["temperature"] * df["vibration"]

    if {"machine_id", "temperature"}.issubset(df.columns):
        machine_temp_mean = df.groupby("machine_id")["temperature"].transform("mean")
        df["temp_diff_from_machine_mean"] = df["temperature"] - machine_temp_mean

    if {"machine_id", "vibration"}.issubset(df.columns):
        machine_vib_mean = df.groupby("machine_id")["vibration"].transform("mean")
        df["vib_diff_from_machine_mean"] = df["vibration"] - machine_vib_mean

    # include column energy_per_vibration for the energy_consumption column
    if {"energy_consumption", "vibration"}.issubset(df.columns):
        df["energy_per_vibration"] = df["energy_consumption"] / (df["vibration"].abs() + eps)

    return df

# below is to be added to verify the data cleaner has cleaned the dataset
def verify_cleaned_data(df: pd.DataFrame, target_column: str = "maintenance_required") -> dict:
    """Run sanity checks to confirm cleaning worked and return a summary report."""
    null_counts = df.isna().sum()
    remaining_nulls = int(null_counts.sum())
    dup_rows = int(df.duplicated().sum())
    has_dup_cols = bool(df.columns.duplicated().any())

    out_of_bounds = {}
    for col, (low, high) in global_bounds.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            bad_low = int((s < low).sum()) if low is not None else 0
            bad_high = int((s > high).sum()) if high is not None else 0
            if bad_low or bad_high:
                out_of_bounds[col] = {"below_min": bad_low, "above_max": bad_high}

    target_ok = True
    if target_column in df.columns:
        target_ok = pd.api.types.is_integer_dtype(df[target_column]) and not df[target_column].isna().any()

    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "remaining_nulls": remaining_nulls,
        "duplicate_rows": dup_rows,
        "duplicate_column_names": has_dup_cols,
        "out_of_bounds": out_of_bounds,
        "target_clean_int": target_ok,
    }

    # report preview section to ensure no errors
    assert remaining_nulls == 0, f"Found {remaining_nulls} remaining nulls!"
    assert dup_rows == 0, f"Found {dup_rows} duplicate rows!"
    assert not has_dup_cols, "Found duplicate column names!"
    assert not out_of_bounds, f"Found out-of-bounds values: {out_of_bounds}"
    assert target_ok, f"Target column '{target_column}' is not clean integer data!"

    return report

# don't forget to input a check function here to ensure
# that the data cleaner did its job and cleaned up all the data
if __name__ == "__main__":
    raw_df = ingest_data(CONFIG["data_path"])
    cleaned_df = clean_data(raw_df, CONFIG['target_column'])
    report = verify_cleaned_data(cleaned_df, CONFIG["target_column"])
    featured_df = engineer_features(cleaned_df, CONFIG.get("use_feature_engineering"))

    # report preview text
    print("Raw shape:", raw_df.shape)
    print("Cleaned shape:", cleaned_df.shape)
    print("Featured shape:", featured_df.shape)
    print("Remaining nulls:", featured_df.isna().sum().sum())
    print("Verification passed, Report is ready for viewing.", report)

    # save cleaned dataset into a new filepath
    featured_df.to_csv(CONFIG["cleaned_output_path"], index=False)
    print(f"Saved cleaned dataset to '{CONFIG['cleaned_output_path']}' ({featured_df.shape})")