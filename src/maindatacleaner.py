import json
import os
import numpy as np
import pandas as pd

CONFIG = {
    "data_path": "data/ProjectData/proj_machine_readings.csv",
    "cleaned_output_path": "data/ProjectData/proj_machine_readings_cleaned.csv",
    "target_column": "maintenance_required",

    # these few lines are added to make this dataset cleaner more versatile
    "group_column": "machine_id",
    "timestamp_column": "timestamp",
    "keep_timestamp": False,
    "drop_columns": [],
    "abs_columns": ["vibration"],
    "imputation_strategy": "machine_median",  # "machine_median", "median", "mean", "zero"
    "integer_cols": ["machine_status", "anomaly_flag", "maintenance_required"],

    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",  # Helpful if maintenance_required=1 is rarer
    "use_feature_engineering": True,

    # feature_engineering configuration is done here
    "feature_engineering": {
        "enabled": True,
        "epsilon": 1e-5,
        "include_temp_pressure_ratio": True,
        "include_thermal_vibration_index": True,
        "include_machine_mean_diffs": True,
        "include_energy_per_vibration": True,
        "custom_ratios": [],
        # Example custom ratio:
        # {"name": "temp_humidity_ratio", "num": "temperature", "den": "humidity"},
        "custom_group_diffs": [],
        # Example custom group diff:
        # {"name": "press_diff_from_machine_mean", "group_col": "machine_id", "col": "pressure", "agg": "mean"},
    },

    "save_model_dir": "saved_model",
    "output_plot_path": "data/feature_importances.png",
}

# these bounds will apply to all datasets unless overridden
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

# this extra definition is to do all imputation 
# before ingesting the data for processing
def _impute_series(df: pd.DataFrame, col: str, strategy: str, group_col: str | None) -> pd.Series:
    s = df[col]
    if not s.isna().any():
        return s

    if strategy == "machine_median" and group_col and group_col in df.columns:
        machine_med = df.groupby(group_col)[col].transform("median")
        s = s.fillna(machine_med)
        if s.isna().any():
            s = s.fillna(s.median())
    elif strategy == "mean":
        s = s.fillna(s.mean())
    elif strategy == "zero":
        s = s.fillna(0.0)
    else:
        s = s.fillna(s.median())

    # safety fallback to account for columns fully populated with NaN
    if s.isna().any():
        s = s.fillna(0.0)
    return s

def ingest_data(csv_path: str) -> pd.DataFrame:
    """Ingest raw CSV data from disk."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at '{csv_path}'.")
    return pd.read_csv(csv_path)

def clean_data(df: pd.DataFrame, target_column: str = "maintenance_required", config: dict | None = None) -> pd.DataFrame:
    """Clean duplicates, types, invalid ranges, and missing values."""

    # added to check for cleaning configurations
    cfg = config if isinstance(config, dict) else {}
    group_col = cfg.get("group_column", "machine_id")
    ts_col = cfg.get("timestamp_column", "timestamp")
    keep_ts = bool(cfg.get("keep_timestamp", False))
    drop_cols = list(cfg.get("drop_columns", []))
    abs_cols = list(cfg.get("abs_columns", ["vibration"]))
    imputation_strategy = cfg.get("imputation_strategy", "machine_median")
    int_cols = list(cfg.get("integer_cols", ["machine_status", "anomaly_flag", "maintenance_required"]))
    bounds = cfg.get("bounds", global_bounds)

    df = df.copy()

    # required line to make certain data values hashable
    # for removal of duplicates
    for col in df.columns:
        if df[col].dtype == object and df[col].map(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].map(_make_hashable)

    df = df.drop_duplicates()

    if ts_col and ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.sort_values(ts_col)
        if not keep_ts:
            df = df.drop(columns=[ts_col])

    existing_drops = [c for c in drop_cols if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)

    if group_col and group_col in df.columns and not pd.api.types.is_integer_dtype(df[group_col]):
        df[group_col] = df[group_col].astype(str).str.strip()

    if "failure_type" in df.columns:
        df["failure_type"] = df["failure_type"].astype(str).str.strip()
        df.loc[df["failure_type"].isin(["", "nan", "None", "null"]), "failure_type"] = "Unknown"

    # additional input here to fix some sign-flipped negative vibration sensor glitching
    for col in abs_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").abs()

    for col, (low, high) in bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if low is not None or high is not None:
                df[col] = df[col].clip(lower=low, upper=high)

    for col in int_cols:
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

    # code here is to fast forward the imputation of values in data cleaning
    skip_impute = {target_column} if isinstance(target_column, str) else set()
    num_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in skip_impute
    ]
    for col in num_cols:
        df[col] = _impute_series(df, col, imputation_strategy, group_col)
    for col in int_cols:
        if col in df.columns and col != target_column and not df[col].isna().any():
            df[col] = df[col].astype(int)

    return df

# the more versatile version of engineering features settings
def engineer_features(df: pd.DataFrame, enabled: bool = True, fe_config: dict | None = None,
    group_col: str = "machine_id",) -> pd.DataFrame:
    """Create domain features from sensor and derived columns."""
    df = df.copy()
    fe_cfg = fe_config if isinstance(fe_config, dict) else {}
    if not enabled or not fe_cfg.get("enabled", True):
        return df

    eps = float(fe_cfg.get("epsilon", 1e-5))

    if fe_cfg.get("include_temp_pressure_ratio", True) and {"temperature", "pressure"}.issubset(df.columns):
        df["temp_pressure_ratio"] = df["temperature"] / (df["pressure"].abs() + eps)

    if fe_cfg.get("include_thermal_vibration_index", True) and {"vibration", "temperature"}.issubset(df.columns):
        df["thermal_vibration_index"] = df["temperature"] * df["vibration"]

    if fe_cfg.get("include_machine_mean_diffs", True) and group_col in df.columns:
        if "temperature" in df.columns:
            machine_temp_mean = df.groupby(group_col)["temperature"].transform("mean")
            df["temp_diff_from_machine_mean"] = df["temperature"] - machine_temp_mean
        if "vibration" in df.columns:
            machine_vib_mean = df.groupby(group_col)["vibration"].transform("mean")
            df["vib_diff_from_machine_mean"] = df["vibration"] - machine_vib_mean

    if fe_cfg.get("include_energy_per_vibration", True) and {"energy_consumption", "vibration"}.issubset(df.columns):
        df["energy_per_vibration"] = df["energy_consumption"] / (df["vibration"].abs() + eps)

    for ratio_spec in fe_cfg.get("custom_ratios", []):
        out_col = ratio_spec.get("name")
        num_col = ratio_spec.get("num")
        den_col = ratio_spec.get("den")
        if out_col and {num_col, den_col}.issubset(df.columns):
            df[out_col] = df[num_col] / (df[den_col].abs() + eps)

    for diff_spec in fe_cfg.get("custom_group_diffs", []):
        out_col = diff_spec.get("name")
        grp = diff_spec.get("group_col", group_col)
        val_col = diff_spec.get("col")
        agg = diff_spec.get("agg", "mean")
        if out_col and {grp, val_col}.issubset(df.columns):
            baseline = df.groupby(grp)[val_col].transform(agg)
            df[out_col] = df[val_col] - baseline

    return df

# below is to be added to verify the data cleaner has cleaned the dataset
# more versatile version of cleaning this dataset
def verify_cleaned_data(df: pd.DataFrame, target_column: str = "maintenance_required", 
                        bounds: dict | None = None,) -> dict:
    """Run sanity checks to confirm cleaning worked and return a summary report."""
    active_bounds = bounds if isinstance(bounds, dict) else global_bounds
    remaining_nulls = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum())
    has_dup_cols = bool(df.columns.duplicated().any())

    out_of_bounds = {}
    for col, (low, high) in active_bounds.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            bad_low = int((s < low).sum()) if low is not None else 0
            bad_high = int((s > high).sum()) if high is not None else 0
            if bad_low or bad_high:
                out_of_bounds[col] = {"below_min": bad_low, "above_max": bad_high}

    target_ok = True
    if target_column in df.columns:
        target_ok = pd.api.types.is_integer_dtype(df[target_column]) and not df[target_column].isna().any()

    # report configuration
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
    out_path = CONFIG.get("cleaned_output_path")
    if out_path:
        parent_dir = os.path.dirname(out_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        featured_df.to_csv(out_path, index=False)
    featured_df.to_csv(CONFIG["cleaned_output_path"], index=False)
    print(f"Saved cleaned dataset to '{CONFIG['cleaned_output_path']}' ({featured_df.shape})")