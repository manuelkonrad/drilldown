#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Manuel Konrad
#
# SPDX-License-Identifier: MIT

"""Module to create synthetic timeseries dataset for demonstration purposes.

This module generates a timeseries dataset spanning from 3 months ago to today,
with various patterns including stable variables, drifting variables, change points,
and outages.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from deltalake import write_deltalake
from PIL import Image


def _create_test_image(
    path: Path,
    width: int = 100,
    height: int = 100,
    color: tuple[int, int, int] = (255, 0, 0),
    pattern: str = "solid",
) -> str:
    """Create a test image file with different patterns.

    Args:
        path: Path to save the image.
        width: Image width.
        height: Image height.
        color: Base color for the image.
        pattern: Pattern type - 'solid', 'gradient', 'checkerboard', 'noise'.

    Returns:
        Path to the saved image as a string.
    """
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    if pattern == "solid":
        img_array[:, :] = color
    elif pattern == "gradient":
        for i in range(width):
            factor = i / width
            img_array[:, i] = [int(c * factor) for c in color]
    elif pattern == "checkerboard":
        square_size = 10
        for i in range(height):
            for j in range(width):
                if (i // square_size + j // square_size) % 2 == 0:
                    img_array[i, j] = color
                else:
                    img_array[i, j] = [255 - c for c in color]
    elif pattern == "noise":
        noise = np.random.randint(-50, 50, (height, width, 3))
        img_array = np.clip(np.array(color) + noise, 0, 255).astype(np.uint8)

    Image.fromarray(img_array).save(path)
    return str(path)


def _create_test_curve_json(
    path: Path, points: int = 10, curve_type: str = "quadratic"
) -> str:
    """Create a test curve JSON file with different patterns.

    Args:
        path: Path to save the curve JSON.
        points: Number of points in the curve.
        curve_type: Type of curve - 'quadratic', 'sine', 'exponential',
            'linear', 'step', 'noisy'.

    Returns:
        Path to the saved curve as a string.
    """
    x = list(range(points))

    if curve_type == "quadratic":
        y = [float(i**2) for i in range(points)]
    elif curve_type == "sine":
        y = [float(np.sin(i * np.pi / 10) * 50 + 50) for i in range(points)]
    elif curve_type == "exponential":
        y = [float(np.exp(i * 0.1)) for i in range(points)]
    elif curve_type == "linear":
        y = [float(i * 2.5) for i in range(points)]
    elif curve_type == "step":
        y = [float(i // 5 * 10) for i in range(points)]
    elif curve_type == "noisy":
        base = np.array([i * 2 for i in range(points)])
        noise = np.random.normal(0, 5, points)
        y = [float(b + n) for b, n in zip(base, noise, strict=True)]
    else:
        y = [float(i**2) for i in range(points)]

    curve_data = {"x": x, "y": y}
    with path.open("w") as f:
        json.dump(curve_data, f)
    return str(path)


def _generate_reusable_assets(
    assets_dir: Path, num_assets: int = 10, seed: int | None = None
) -> tuple[list[str], list[str]]:
    """Generate a limited set of reusable images and curves.

    Args:
        assets_dir: Directory to store assets.
        num_assets: Number of unique assets to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (image_paths, curve_paths).
    """
    if seed is not None:
        np.random.seed(seed)

    images_dir = assets_dir / "images"
    images_dir.mkdir(exist_ok=True)
    curves_dir = assets_dir / "curves"
    curves_dir.mkdir(exist_ok=True)

    patterns = ["solid", "gradient", "checkerboard", "noise"]
    curve_types = ["quadratic", "sine", "exponential", "linear", "step", "noisy"]

    image_paths = []
    curve_paths = []

    for i in range(num_assets):
        pattern = patterns[i % len(patterns)]
        color = ((i * 30) % 256, (i * 60) % 256, (i * 90) % 256)
        img_path = images_dir / f"image_{i:02d}.png"
        _create_test_image(
            img_path, width=100, height=100, color=color, pattern=pattern
        )
        image_paths.append(str(img_path))

        curve_type = curve_types[i % len(curve_types)]
        curve_path = curves_dir / f"curve_{i:02d}.json"
        _create_test_curve_json(curve_path, points=20 + (i % 10), curve_type=curve_type)
        curve_paths.append(str(curve_path))

    return image_paths, curve_paths


def _add_drift(
    values: np.ndarray, drift_start_idx: int, drift_rate: float = 0.01
) -> np.ndarray:
    """Add a slow drift to values starting at a specific index.

    Args:
        values: Array of values to modify.
        drift_start_idx: Index where drift starts.
        drift_rate: Rate of drift per sample.

    Returns:
        Modified array with drift.
    """
    result = values.copy()
    drift_length = len(values) - drift_start_idx
    if drift_length > 0:
        drift = np.arange(drift_length) * drift_rate
        result[drift_start_idx:] += drift
    return result


def _add_change_point(
    values: np.ndarray, change_idx: int, change_magnitude: float
) -> np.ndarray:
    """Add a sudden change point to values at a specific index.

    Args:
        values: Array of values to modify.
        change_idx: Index where change occurs.
        change_magnitude: Magnitude of the change.

    Returns:
        Modified array with change point.
    """
    result = values.copy()
    result[change_idx:] += change_magnitude
    return result


def _add_outage(
    values: np.ndarray,
    outage_start_idx: int,
    outage_duration: int,
    outage_value: float = 0.0,
) -> np.ndarray:
    """Add an outage period to values.

    Args:
        values: Array of values to modify.
        outage_start_idx: Index where outage starts.
        outage_duration: Duration of outage in samples.
        outage_value: Value during outage.

    Returns:
        Modified array with outage.
    """
    result = values.copy()
    outage_end = min(outage_start_idx + outage_duration, len(values))
    result[outage_start_idx:outage_end] = outage_value
    return result


def _generate_production_line_data(
    num_parts: int = 100,
    num_steps: int = 5,
    base_date: datetime | None = None,
    step_duration_mean: float = 72.0,
    step_duration_std: float = 12.0,
    setup_time_mean: float = 5.0,
    setup_time_std: float = 1.0,
    outlier_probability: float = 0.02,
    outlier_delay_hours: tuple[float, float] = (4.0, 12.0),
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic production line data with multiple timestamps.

    Generates one part per hour over the specified time range, with cycle times
    around 6 hours total and occasional outliers where a part gets stuck.

    Args:
        num_parts: Number of parts to generate.
        num_steps: Number of process steps in the production line.
        base_date: Starting date for the production line.
        step_duration_mean: Mean duration for each step in minutes.
        step_duration_std: Standard deviation for step duration in minutes.
        setup_time_mean: Mean setup/transition time between parts in minutes.
        setup_time_std: Standard deviation for setup time in minutes.
        outlier_probability: Probability of a part getting stuck at a step.
        outlier_delay_hours: Range of additional delay hours for outliers.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with part_id, timestamps, cycle times, and outlier flags.
    """
    if seed is not None:
        np.random.seed(seed)

    if base_date is None:
        base_date = datetime.now().replace(
            minute=0, second=0, microsecond=0
        ) - timedelta(days=90)

    data: dict[str, list] = {
        "part_id": [f"PART_{i:05d}" for i in range(num_parts)],
    }
    step_available_times = [base_date for _ in range(num_steps)]
    outlier_flags: list[bool] = []

    for part_idx in range(num_parts):
        part_timestamps: list[datetime] = []
        is_outlier = False

        for step_idx in range(num_steps):
            if step_idx == 0:
                if part_idx > 0:
                    setup_time = max(
                        0,
                        np.random.normal(setup_time_mean, setup_time_std),
                    )
                    step_available_times[0] += timedelta(minutes=setup_time)
                start_time = step_available_times[0]
            else:
                part_arrival = part_timestamps[step_idx - 1]
                step_available = step_available_times[step_idx]
                start_time = max(part_arrival, step_available)

            duration = max(1.0, np.random.normal(step_duration_mean, step_duration_std))

            if np.random.random() < outlier_probability:
                delay_hours = np.random.uniform(
                    outlier_delay_hours[0], outlier_delay_hours[1]
                )
                duration += delay_hours * 60
                is_outlier = True

            end_time = start_time + timedelta(minutes=duration)
            step_available_times[step_idx] = end_time
            part_timestamps.append(end_time)

        for step_idx, timestamp in enumerate(part_timestamps):
            col_name = f"step_{step_idx + 1}_timestamp"
            if col_name not in data:
                data[col_name] = []
            data[col_name].append(timestamp)

        outlier_flags.append(is_outlier)

    data["timestamp"] = data["step_1_timestamp"].copy()
    data["is_outlier"] = outlier_flags

    # Calculate cycle times between steps
    for step_idx in range(1, num_steps):
        prev_col = f"step_{step_idx}_timestamp"
        curr_col = f"step_{step_idx + 1}_timestamp"
        cycle_col = f"cycle_time_step_{step_idx}_to_{step_idx + 1}"
        data[cycle_col] = [
            (curr - prev).total_seconds() / 60.0
            for prev, curr in zip(data[prev_col], data[curr_col], strict=True)
        ]

    data["total_cycle_time"] = [
        (
            data[f"step_{num_steps}_timestamp"][i] - data["step_1_timestamp"][i]
        ).total_seconds()
        / 60.0
        for i in range(num_parts)
    ]

    return pd.DataFrame(data)


def _generate_timestamps_from_multistep(
    multistep_df: pd.DataFrame,
    num_steps: int,
    seed: int | None = None,
) -> tuple[list[datetime], dict[str, list]]:
    """Extract timestamps and related data from multistep data.

    Args:
        multistep_df: DataFrame from multistep production data.
        num_steps: Number of steps in the multistep process.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (timestamps, multistep_data dict).
    """
    if seed is not None:
        np.random.seed(seed)

    if multistep_df.empty:
        return [], {}

    timestamps = multistep_df["timestamp"].tolist()
    multistep_data: dict[str, list] = {}

    # Add step timestamps
    for step_idx in range(1, num_steps + 1):
        col = f"step_{step_idx}_timestamp"
        if col in multistep_df.columns:
            multistep_data[col] = multistep_df[col].tolist()

    # Add cycle times between steps
    for step_idx in range(1, num_steps):
        col = f"cycle_time_step_{step_idx}_to_{step_idx + 1}"
        if col in multistep_df.columns:
            multistep_data[col] = multistep_df[col].tolist()

    # Add other columns
    for col in ["total_cycle_time", "is_outlier", "part_id"]:
        if col in multistep_df.columns:
            multistep_data[col] = multistep_df[col].tolist()

    return timestamps, multistep_data


def _create_synthetic_timeseries(
    output_dir: str,
    reusable_image_paths: list[str] | None = None,
    reusable_curve_paths: list[str] | None = None,
    seed: int | None = None,
    multistep_df: pd.DataFrame | None = None,
    num_steps: int = 5,
) -> str:
    """Create a synthetic timeseries dataset with numerical and categorical columns.

    Args:
        output_dir: Directory where the Delta table will be created.
        reusable_image_paths: List of image paths to reuse.
        reusable_curve_paths: List of curve paths to reuse.
        seed: Random seed for reproducibility.
        multistep_df: DataFrame from multistep production data.
        num_steps: Number of steps in the multistep process.

    Returns:
        Path to the created Delta table.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create Delta table directory
    table_dir = output_path / "synthetic_timeseries"

    # Generate time series data
    if multistep_df is not None:
        # Use timestamps from multistep data (one sample per part)
        timestamps, multistep_data = _generate_timestamps_from_multistep(
            multistep_df, num_steps, seed
        )
        num_rows = len(timestamps)
        samples_per_day = 24  # Roughly one sample per hour
    else:
        # Generate timestamps from 3 months ago to today
        end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=90)

        samples_per_day = 24  # One sample per hour
        num_days = 90
        num_rows = num_days * samples_per_day

        # Generate timestamps - one per hour
        timestamps = [start_date + timedelta(hours=i) for i in range(num_rows)]
        multistep_data = None

    # Initialize random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Generate numerical columns with different patterns

    # 1. Temperature: Stable
    temperature_base = 22.0
    temperature_noise = 0.5
    temperature = np.random.normal(temperature_base, temperature_noise, num_rows)

    # 2. Pressure: Stable
    pressure_base = 101.3
    pressure_noise = 0.3
    pressure = np.random.normal(pressure_base, pressure_noise, num_rows)

    # 3. Flow rate: Drifting (slow increase starting at day 20)
    flow_base = 50.0
    flow_noise = 2.0
    flow_rate = np.random.normal(flow_base, flow_noise, num_rows)
    flow_rate = _add_drift(
        flow_rate, drift_start_idx=20 * samples_per_day, drift_rate=0.003
    )

    # 4. Vibration: Drifting (gradual increase from start)
    vibration_base = 0.5
    vibration_noise = 0.05
    vibration = np.random.normal(vibration_base, vibration_noise, num_rows)
    vibration = _add_drift(vibration, drift_start_idx=0, drift_rate=0.0003)

    # 5. Power consumption: Change point at day 30
    power_base = 150.0
    power_noise = 5.0
    power = np.random.normal(power_base, power_noise, num_rows)
    power = _add_change_point(
        power, change_idx=30 * samples_per_day, change_magnitude=15.0
    )

    # 6. Efficiency: Change point at day 45
    efficiency_base = 0.95
    efficiency_noise = 0.01
    efficiency = np.random.normal(efficiency_base, efficiency_noise, num_rows)
    efficiency = _add_change_point(
        efficiency, change_idx=45 * samples_per_day, change_magnitude=-0.08
    )
    efficiency = np.clip(efficiency, 0.0, 1.0)

    # 7. pH level: Outage from day 60 to day 65 (equipment failure)
    ph_base = 7.0
    ph_noise = 0.1
    ph = np.random.normal(ph_base, ph_noise, num_rows)
    ph = _add_outage(
        ph,
        outage_start_idx=60 * samples_per_day,
        outage_duration=5 * samples_per_day,
        outage_value=0.0,
    )

    # 8. Concentration: Outage from day 70 to day 72 (sensor malfunction)
    concentration_base = 10.0
    concentration_noise = 0.3
    concentration = np.random.normal(concentration_base, concentration_noise, num_rows)
    concentration = _add_outage(
        concentration,
        outage_start_idx=70 * samples_per_day,
        outage_duration=2 * samples_per_day,
        outage_value=0.0,
    )

    # Generate categorical columns

    # Machine ID - 5 different machines
    machine_ids = [f"MACHINE_{i % 5 + 1:02d}" for i in range(num_rows)]

    # Operator shift - 3 shifts per day
    shifts = []
    for i in range(num_rows):
        hour = (i % samples_per_day) * 24 / samples_per_day
        if hour < 8:
            shift = "night"
        elif hour < 16:
            shift = "morning"
        else:
            shift = "afternoon"
        shifts.append(shift)

    # Product type - 3 product types with different distributions over time
    product_types = []
    early_period_samples = min(10 * samples_per_day, num_rows)
    middle_period_end = min(20 * samples_per_day, num_rows)
    middle_period_samples = max(0, middle_period_end - early_period_samples)
    late_period_samples = max(0, num_rows - middle_period_end)

    # Generate all product types at once for efficiency
    if early_period_samples > 0:
        product_types.extend(
            np.random.choice(
                ["Type_A", "Type_B", "Type_C"],
                size=early_period_samples,
                p=[0.7, 0.2, 0.1],
            ).tolist()
        )
    if middle_period_samples > 0:
        product_types.extend(
            np.random.choice(
                ["Type_A", "Type_B", "Type_C"],
                size=middle_period_samples,
                p=[0.4, 0.4, 0.2],
            ).tolist()
        )
    if late_period_samples > 0:
        product_types.extend(
            np.random.choice(
                ["Type_A", "Type_B", "Type_C"],
                size=late_period_samples,
                p=[0.2, 0.6, 0.2],
            ).tolist()
        )

    # Maintenance status - changes over time (every 7th day starting from day 0)
    maintenance_status = []
    for i in range(num_rows):
        day = i // samples_per_day
        if day % 7 == 0:  # Maintenance every 7th day (day 0, 7, 14, 21, etc.)
            status = "maintenance"
        elif day % 7 == 1:
            status = "post_maintenance"
        else:
            status = "normal"
        maintenance_status.append(status)

    # Generate quality label based on the numerical features
    # Quality is "good" when all parameters are in acceptable ranges, otherwise "defect"
    # Also create a boolean quality_gate variable
    quality_labels = []
    quality_gate_values = []
    for i in range(num_rows):
        # Define acceptable ranges (these shift with the change points)
        temp_ok = 20.0 <= temperature[i] <= 25.0
        pressure_ok = 99.0 <= pressure[i] <= 105.0
        flow_ok = 40.0 <= flow_rate[i] <= 55.0
        vibration_ok = vibration[i] <= 0.7
        power_ok = 140.0 <= power[i] <= 180.0
        efficiency_ok = efficiency[i] >= 0.85
        ph_ok = 6.5 <= ph[i] <= 7.8
        concentration_ok = 9.0 <= concentration[i] <= 11.0

        # Count how many parameters are out of range
        checks = [
            temp_ok,
            pressure_ok,
            flow_ok,
            vibration_ok,
            power_ok,
            efficiency_ok,
            ph_ok,
            concentration_ok,
        ]
        num_ok = sum(checks)

        # Quality classification
        if num_ok >= 7:  # At least 7 out of 8 checks pass
            quality = "good"
            quality_gate = True
        elif num_ok >= 5:
            quality = "acceptable"
            quality_gate = False
        else:
            quality = "defect"
            quality_gate = False

        quality_labels.append(quality)
        quality_gate_values.append(quality_gate)

    # Generate image and curve URI columns by reusing assets
    image_uris: list[str | None] = []
    curve_uris: list[str | None] = []
    if reusable_image_paths and reusable_curve_paths:
        num_images = len(reusable_image_paths)
        num_curves = len(reusable_curve_paths)
        for i in range(num_rows):
            image_uris.append(reusable_image_paths[i % num_images])
            curve_uris.append(reusable_curve_paths[i % num_curves])
    else:
        image_uris = [None] * num_rows
        curve_uris = [None] * num_rows

    # Create DataFrame
    data = {
        # PRIMARY_ID
        "sample_id": [f"SAMPLE_{i:06d}" for i in range(num_rows)],
        # PRIMARY_TIMESTAMP
        "timestamp": timestamps,
        # NUMERICAL COLUMNS
        "temperature": temperature,
        "pressure": pressure,
        "flow_rate": flow_rate,
        "vibration": vibration,
        "power_consumption": power,
        "efficiency": efficiency,
        "ph_level": ph,
        "concentration": concentration,
        # CATEGORICAL COLUMNS
        "machine_id": machine_ids,
        "shift": shifts,
        "product_type": product_types,
        "maintenance_status": maintenance_status,
        "quality": quality_labels,
        "quality_gate": quality_gate_values,
        # URI COLUMNS
        "image_uri": image_uris,
        "curve_uri": curve_uris,
        # DATE (for partitioning)
        "date": [ts.date() for ts in timestamps],
        # YEAR_WEEK (format: YYYY-cwWW)
        "year_week": [f"{ts.year}-cw{ts.isocalendar()[1]:02d}" for ts in timestamps],
        # YEAR_MONTH (format: YYYY-MM)
        "year_month": [f"{ts.year}-{ts.month:02d}" for ts in timestamps],
    }

    # Add multistep data (step timestamps, cycle times, etc.) if available
    if multistep_data is not None:
        for col_name, values in multistep_data.items():
            if values:  # Only add if there are values
                data[col_name] = values

    df = pd.DataFrame(data)

    # Define field metadata for proper column type inference
    field_metadata = {
        # Primary fields
        "sample_id": {
            "column_type": "primary_id",
            "description": "Primary sample identifier",
        },
        "timestamp": {
            "column_type": "primary_timestamp",
            "description": "Primary timestamp",
        },
        # Numerical fields
        "temperature": {
            "column_type": "numerical",
            "description": "Temperature in Celsius (stable)",
        },
        "pressure": {
            "column_type": "numerical",
            "description": "Pressure in kPa (stable)",
        },
        "flow_rate": {
            "column_type": "numerical",
            "description": "Flow rate in L/min (drifting from day 20)",
        },
        "vibration": {
            "column_type": "numerical",
            "description": "Vibration amplitude in mm (drifting from start)",
        },
        "power_consumption": {
            "column_type": "numerical",
            "description": "Power consumption in W (change point at day 30)",
        },
        "efficiency": {
            "column_type": "numerical",
            "description": "Process efficiency ratio (change point at day 45)",
        },
        "ph_level": {
            "column_type": "numerical",
            "description": "pH level (outage days 60-65)",
        },
        "concentration": {
            "column_type": "numerical",
            "description": "Concentration in mg/L (outage days 70-72)",
        },
        # Categorical fields
        "machine_id": {
            "column_type": "categorical",
            "description": "Machine identifier",
        },
        "shift": {
            "column_type": "categorical",
            "description": "Operator shift (night/morning/afternoon)",
        },
        "product_type": {
            "column_type": "categorical",
            "description": "Product type classification",
        },
        "maintenance_status": {
            "column_type": "categorical",
            "description": "Maintenance status",
        },
        "quality": {
            "column_type": "categorical",
            "description": "Final quality label (good/acceptable/defect)",
        },
        "quality_gate": {
            "column_type": "categorical",
            "description": "Quality gate pass/fail (True for good quality, False otherwise)",
        },
        # URI fields
        "image_uri": {
            "column_type": "uri_img",
            "description": "Image file URI (reused from 10 unique images)",
        },
        "curve_uri": {
            "column_type": "uri_curve",
            "object_format": "json",
            "description": "Curve file URI (reused from 10 unique curves)",
        },
        # Date fields
        "date": {"column_type": "date", "description": "Sample date"},
        "year_week": {"column_type": "year_week", "description": "Year and week"},
        "year_month": {"column_type": "year_month", "description": "Year and month"},
    }

    # Add metadata for multistep columns if present
    if multistep_data is not None:
        for col_name in multistep_data.keys():
            if col_name in df.columns:
                if col_name == "total_cycle_time":
                    field_metadata[col_name] = {
                        "column_type": "numerical",
                        "description": "Total cycle time from first to last step in minutes",
                    }
                elif col_name == "is_outlier":
                    field_metadata[col_name] = {
                        "column_type": "categorical",
                        "description": "Whether the part got stuck at a process step (outlier)",
                    }
                elif col_name == "part_id":
                    field_metadata[col_name] = {
                        "column_type": "categorical",
                        "description": "Part identifier from multistep process",
                    }
                elif col_name.startswith("step_") and col_name.endswith("_timestamp"):
                    field_metadata[col_name] = {
                        "column_type": "timestamp",
                        "description": "Timestamp for completion of process step",
                    }
                elif col_name.startswith("cycle_time_step_"):
                    field_metadata[col_name] = {
                        "column_type": "numerical",
                        "description": "Cycle time between steps in minutes",
                    }

    # Convert DataFrame to PyArrow table with metadata
    # Create schema with metadata
    fields = []
    for col in df.columns:
        arrow_type = pa.Schema.from_pandas(df[[col]]).field(0).type
        metadata = field_metadata.get(col, {})
        # Convert metadata dict to bytes
        metadata_bytes = {k.encode(): v.encode() for k, v in metadata.items()}
        fields.append(pa.field(col, arrow_type, metadata=metadata_bytes))

    schema = pa.schema(fields)
    table = pa.Table.from_pandas(df, schema=schema)

    # Write to Delta table with partitioning
    write_deltalake(
        str(table_dir),
        table,
        mode="overwrite",
        partition_by=["date", "year_week", "year_month"],
        description="Synthetic timeseries dataset for demonstration purposes",
    )

    return str(table_dir)


def create_synthetic_dataset(
    output_dir: str,
    num_reusable_assets: int = 10,
    num_steps: int = 5,
    seed: int = 42,
) -> str:
    """Create synthetic timeseries dataset for demonstration purposes.

    Creates a complete demo dataset spanning 90 days with:
    - Numerical columns with various patterns (stable, drifting, change points, outages)
    - Categorical columns (machine_id, shift, product_type, maintenance_status, quality)
    - Multistep production timestamps and cycle times
    - Reusable image and curve assets

    Args:
        output_dir: Directory where the dataset will be created.
        num_reusable_assets: Number of unique images/curves to generate.
        num_steps: Number of steps in the multistep process.
        seed: Random seed for reproducibility.

    Returns:
        Path to the created dataset.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    assets_dir = output_path / "assets"
    assets_dir.mkdir(exist_ok=True)
    image_paths, curve_paths = _generate_reusable_assets(
        assets_dir, num_assets=num_reusable_assets, seed=seed
    )

    # Generate production line data (one part per hour for 90 days)
    base_date = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(
        days=90
    )
    num_parts = 90 * 24
    multistep_df = _generate_production_line_data(
        num_parts=num_parts,
        num_steps=num_steps,
        base_date=base_date,
        step_duration_mean=72.0,
        step_duration_std=12.0,
        setup_time_mean=5.0,
        setup_time_std=1.0,
        outlier_probability=0.02,
        outlier_delay_hours=(4.0, 12.0),
        seed=seed,
    )

    return _create_synthetic_timeseries(
        output_dir=output_dir,
        reusable_image_paths=image_paths,
        reusable_curve_paths=curve_paths,
        seed=seed,
        multistep_df=multistep_df,
        num_steps=num_steps,
    )
