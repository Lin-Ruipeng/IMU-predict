from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from imup.angle_extrema.core import AngleExtremaError
from imup.preprocessing import find_repository_root

FloatArray = npt.NDArray[np.float64]

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_AXES = ("x", "y", "z")
_ALLOWED_CONFIG_KEYS = frozenset(
    {
        "comparison_name",
        "analysis_run_name",
        "target_cutoff_hz",
        "comparison_cutoff_frequencies_hz",
        "include_raw",
        "force",
    }
)


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration for target-integrated-angle comparisons."""

    comparison_name: str
    analysis_run_name: str
    target_cutoff_hz: float = 11.0
    comparison_cutoff_frequencies_hz: tuple[float, ...] = (3.0, 5.0, 7.0, 9.0)
    include_raw: bool = True
    force: bool = False


@dataclass(frozen=True)
class ComparisonDiagnostic:
    """One path-scoped comparison warning or failure."""

    path: Path | None
    message: str


@dataclass(frozen=True)
class ComparisonRunResult:
    """Observable result of one complete comparison attempt."""

    output_dir: Path
    processed_sources: int
    comparisons: int
    failures: tuple[ComparisonDiagnostic, ...]
    warnings: tuple[ComparisonDiagnostic, ...]

    @property
    def exit_code(self) -> int:
        return int(bool(self.failures))


@dataclass(frozen=True)
class _AngleData:
    rows: FloatArray
    sample_interval_s: float


def load_comparison_config(
    config_path: Path,
    *,
    repository_root: Path | None = None,
) -> ComparisonConfig:
    """Load and strictly validate one comparison JSON configuration."""
    root = (repository_root or find_repository_root()).resolve()
    resolved_path = _resolve_path(config_path, root)
    try:
        with resolved_path.open(encoding="utf-8") as config_file:
            payload: object = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise AngleExtremaError(
            f"could not read comparison JSON: {error}", path=resolved_path
        ) from error
    if not isinstance(payload, dict):
        raise AngleExtremaError(
            "comparison JSON must contain one object", path=resolved_path
        )
    if not all(isinstance(key, str) for key in payload):
        raise AngleExtremaError(
            "comparison JSON keys must be strings", path=resolved_path
        )
    values = cast(Mapping[str, object], payload)
    unknown = set(values) - _ALLOWED_CONFIG_KEYS
    if unknown:
        raise AngleExtremaError(
            f"unknown comparison JSON keys: {', '.join(sorted(unknown))}",
            path=resolved_path,
        )
    comparison_name = values.get("comparison_name")
    analysis_run_name = values.get("analysis_run_name")
    if not isinstance(comparison_name, str) or not comparison_name:
        raise AngleExtremaError(
            "comparison_name must be a non-empty string", path=resolved_path
        )
    if not isinstance(analysis_run_name, str) or not analysis_run_name:
        raise AngleExtremaError(
            "analysis_run_name must be a non-empty string", path=resolved_path
        )
    cutoffs = _float_list(
        values.get("comparison_cutoff_frequencies_hz", [3.0, 5.0, 7.0, 9.0]),
        "comparison_cutoff_frequencies_hz",
        resolved_path,
    )
    return ComparisonConfig(
        comparison_name=comparison_name,
        analysis_run_name=analysis_run_name,
        target_cutoff_hz=_number(
            values.get("target_cutoff_hz", 11.0),
            "target_cutoff_hz",
            resolved_path,
        ),
        comparison_cutoff_frequencies_hz=cutoffs,
        include_raw=_boolean(
            values.get("include_raw", True), "include_raw", resolved_path
        ),
        force=_boolean(values.get("force", False), "force", resolved_path),
    )


def run_angle_comparison(
    config: ComparisonConfig,
    *,
    repository_root: Path | None = None,
) -> ComparisonRunResult:
    """Compare angle variants with the target and atomically publish plots."""
    root = (repository_root or find_repository_root()).resolve()
    _validate_config(config)
    source_run = root / "outputs" / "angle_extrema" / "runs" / config.analysis_run_name
    output_dir = (
        root / "outputs" / "angle_extrema" / "comparisons" / config.comparison_name
    )
    if not source_run.is_dir():
        raise AngleExtremaError(
            "analysis run directory does not exist", path=source_run
        )
    if output_dir.exists() and not config.force:
        raise AngleExtremaError(
            "comparison directory already exists; rerun with force=true to replace it",
            path=output_dir,
        )
    sources = _analysis_sources(source_run, root)
    if not sources:
        raise AngleExtremaError(
            "analysis run contains no resolved input files", path=source_run
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{config.comparison_name}.", dir=output_dir.parent)
    )
    records: list[dict[str, object]] = []
    failures: list[ComparisonDiagnostic] = []
    warnings: list[ComparisonDiagnostic] = []
    processed_sources = 0
    try:
        _write_json(
            temporary_dir / "comparison_config.json",
            _normalized_config(config),
        )
        for source in sources:
            source_records = 0
            source_run_dir = _source_output_dir(source, source_run, root)
            source_output = _source_output_dir(source, temporary_dir, root)
            try:
                raw = _read_angle_data(source)
                target_variant = f"lowpass_{_cutoff_slug(config.target_cutoff_hz)}hz"
                target_path = source_run_dir / "filtered" / f"{target_variant}.csv"
                target = _read_angle_data(target_path)
                _require_shared_time_axis(raw, target, source, target_path)
                comparators: list[tuple[str, _AngleData, Path]] = []
                if config.include_raw:
                    comparators.append(("raw", raw, source))
                for cutoff_hz in config.comparison_cutoff_frequencies_hz:
                    variant = f"lowpass_{_cutoff_slug(cutoff_hz)}hz"
                    variant_path = source_run_dir / "filtered" / f"{variant}.csv"
                    try:
                        comparator = _read_angle_data(variant_path)
                        _require_shared_time_axis(
                            target, comparator, target_path, variant_path
                        )
                    except (AngleExtremaError, OSError, ValueError) as error:
                        failures.append(
                            ComparisonDiagnostic(
                                path=getattr(error, "path", None) or variant_path,
                                message=str(error),
                            )
                        )
                        continue
                    comparators.append((variant, comparator, variant_path))

                for variant, comparator, _ in comparators:
                    shift_s, correlation = _estimate_shared_shift(
                        target.rows[:, :3],
                        comparator.rows[:, :3],
                        target.sample_interval_s,
                    )
                    aligned_time, aligned_target, aligned_comparator = (
                        _align_comparator(
                            target.rows,
                            comparator.rows,
                            shift_s,
                        )
                    )
                    y_limits = _axis_limits(target.rows[:, :3], comparator.rows[:, :3])
                    prefix = (
                        f"{variant}_vs_target_{_cutoff_slug(config.target_cutoff_hz)}hz"
                    )
                    _plot_comparison(
                        source_output / f"{prefix}_without_delay_compensation.png",
                        target.rows[:, 3],
                        target.rows[:, :3],
                        comparator.rows[:, :3],
                        target_label=f"Target Integrated Angle ({config.target_cutoff_hz:g} Hz)",
                        comparator_label=_variant_label(variant),
                        title="Without delay compensation",
                        y_limits=y_limits,
                    )
                    _plot_comparison(
                        source_output / f"{prefix}_with_delay_compensation.png",
                        aligned_time,
                        aligned_target,
                        aligned_comparator,
                        target_label=f"Target Integrated Angle ({config.target_cutoff_hz:g} Hz)",
                        comparator_label=_variant_label(variant),
                        title="With estimated delay compensation",
                        y_limits=y_limits,
                    )
                    records.append(
                        {
                            "source_file": _display_path(source, root),
                            "comparator": variant,
                            "target": target_variant,
                            "estimated_shift_s": _clean_float(shift_s),
                            "estimated_shift_ms": _clean_float(shift_s * 1000.0),
                            "shift_action": (
                                "delay"
                                if shift_s > 0
                                else "advance"
                                if shift_s < 0
                                else "none"
                            ),
                            "correlation": _clean_float(correlation),
                            "overlap_start_s": _clean_float(float(aligned_time[0])),
                            "overlap_end_s": _clean_float(float(aligned_time[-1])),
                        }
                    )
                    source_records += 1
                if source_records:
                    processed_sources += 1
            except (AngleExtremaError, OSError, ValueError) as error:
                failures.append(
                    ComparisonDiagnostic(
                        path=getattr(error, "path", None) or source,
                        message=str(error),
                    )
                )
            if source_records == 0 and not failures:
                warnings.append(
                    ComparisonDiagnostic(source, "source produced no plots")
                )

        _write_compensation_csv(temporary_dir / "delay_compensation.csv", records)
        _write_readme(temporary_dir / "README.md", config)
        _write_json(
            temporary_dir / "comparison_results.json",
            {
                "status": "failed" if failures else "succeeded",
                "processed_sources": processed_sources,
                "comparisons": len(records),
                "failures": [
                    {
                        "path": _display_path(item.path, root)
                        if item.path is not None
                        else None,
                        "message": item.message,
                    }
                    for item in failures
                ],
                "warnings": [
                    {
                        "path": _display_path(item.path, root)
                        if item.path is not None
                        else None,
                        "message": item.message,
                    }
                    for item in warnings
                ],
            },
        )
        _publish(temporary_dir, output_dir, force=config.force)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return ComparisonRunResult(
        output_dir=output_dir,
        processed_sources=processed_sources,
        comparisons=len(records),
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


def _validate_config(config: ComparisonConfig) -> None:
    for name, value in (
        ("comparison_name", config.comparison_name),
        ("analysis_run_name", config.analysis_run_name),
    ):
        if not _NAME.fullmatch(value):
            raise AngleExtremaError(
                f"{name} must start with an ASCII letter or number and may then "
                "contain letters, numbers, '.', '_' and '-'"
            )
    if not math.isfinite(config.target_cutoff_hz) or config.target_cutoff_hz <= 0:
        raise AngleExtremaError("target_cutoff_hz must be finite and greater than zero")
    cutoffs = config.comparison_cutoff_frequencies_hz
    if len(set(cutoffs)) != len(cutoffs):
        raise AngleExtremaError(
            "comparison_cutoff_frequencies_hz must not contain duplicates"
        )
    if any(not math.isfinite(value) or value <= 0 for value in cutoffs):
        raise AngleExtremaError(
            "comparison_cutoff_frequencies_hz values must be finite and greater than zero"
        )
    if config.target_cutoff_hz in cutoffs:
        raise AngleExtremaError(
            "comparison_cutoff_frequencies_hz must not contain target_cutoff_hz"
        )
    if not config.include_raw and not cutoffs:
        raise AngleExtremaError("at least one comparison variant is required")
    for boolean_name, boolean_value in (
        ("include_raw", config.include_raw),
        ("force", config.force),
    ):
        if not isinstance(boolean_value, bool):
            raise AngleExtremaError(f"{boolean_name} must be a boolean")


def _analysis_sources(source_run: Path, root: Path) -> list[Path]:
    config_path = source_run / "analysis_config.json"
    try:
        payload: object = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AngleExtremaError(
            f"could not read source analysis configuration: {error}", path=config_path
        ) from error
    if not isinstance(payload, dict) or not isinstance(
        payload.get("input_paths"), list
    ):
        raise AngleExtremaError(
            "source analysis configuration has invalid input_paths", path=config_path
        )
    sources: list[Path] = []
    for value in payload["input_paths"]:
        if not isinstance(value, str) or not value:
            raise AngleExtremaError(
                "source analysis input_paths must contain non-empty strings",
                path=config_path,
            )
        resolved = _resolve_path(Path(value), root)
        if resolved.is_file():
            sources.append(resolved)
        elif resolved.is_dir():
            sources.extend(
                sorted(
                    path
                    for path in resolved.iterdir()
                    if path.is_file() and path.name.endswith("integrated_angle.csv")
                )
            )
        else:
            raise AngleExtremaError("source input path does not exist", path=resolved)
    return sources


def _read_angle_data(source: Path) -> _AngleData:
    rows: list[list[float]] = []
    try:
        with source.open(encoding="utf-8", newline="") as csv_file:
            for line_number, row in enumerate(csv.reader(csv_file), start=1):
                if len(row) != 4:
                    raise AngleExtremaError(
                        f"line {line_number} has {len(row)} columns; expected exactly 4",
                        path=source,
                    )
                try:
                    parsed = [float(value) for value in row]
                except ValueError as error:
                    raise AngleExtremaError(
                        f"line {line_number} contains a non-numeric value", path=source
                    ) from error
                if not all(math.isfinite(value) for value in parsed):
                    raise AngleExtremaError(
                        f"line {line_number} contains a non-finite value", path=source
                    )
                rows.append(parsed)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise AngleExtremaError(
            f"could not read angle CSV: {error}", path=source
        ) from error
    if len(rows) < 3:
        raise AngleExtremaError("at least three angle rows are required", path=source)
    array = np.asarray(rows, dtype=np.float64)
    elapsed = array[:, 3]
    intervals = np.diff(elapsed)
    if elapsed[0] != 0.0 or np.any(intervals <= 0):
        raise AngleExtremaError(
            "elapsed_s must start at zero and be strictly increasing", path=source
        )
    interval = float(np.median(intervals))
    if not np.allclose(intervals, interval, rtol=1e-6, atol=1e-12):
        raise AngleExtremaError(
            "elapsed_s intervals must be fixed within floating-point tolerance",
            path=source,
        )
    return _AngleData(array, interval)


def _require_shared_time_axis(
    left: _AngleData,
    right: _AngleData,
    left_path: Path,
    right_path: Path,
) -> None:
    if left.rows.shape != right.rows.shape or not np.array_equal(
        left.rows[:, 3], right.rows[:, 3]
    ):
        raise AngleExtremaError(
            f"time axis does not match {left_path}", path=right_path
        )


def _estimate_shared_shift(
    target: FloatArray,
    comparator: FloatArray,
    sample_interval_s: float,
) -> tuple[float, float]:
    target_diff = np.diff(target, axis=0)
    comparator_diff = np.diff(comparator, axis=0)
    target_scale = np.std(target_diff, axis=0)
    comparator_scale = np.std(comparator_diff, axis=0)
    informative = (target_scale > 1e-12) & (comparator_scale > 1e-12)
    if not np.any(informative):
        raise AngleExtremaError(
            "at least one axis must vary to estimate a shared delay"
        )
    target_diff = _standardize_axes(target_diff[:, informative])
    comparator_diff = _standardize_axes(comparator_diff[:, informative])
    maximum = min(
        int(round(0.5 / sample_interval_s)),
        max(1, target_diff.shape[0] // 5),
    )
    lags = np.arange(-maximum, maximum + 1)
    correlations = np.asarray(
        [_correlation_at_shift(target_diff, comparator_diff, int(lag)) for lag in lags]
    )
    peak = int(np.nanargmax(correlations))
    fractional_lag = float(lags[peak])
    if 0 < peak < correlations.size - 1:
        left, center, right = correlations[peak - 1 : peak + 2]
        denominator = left - 2.0 * center + right
        if denominator != 0.0:
            offset = 0.5 * (left - right) / denominator
            fractional_lag += float(np.clip(offset, -0.5, 0.5))
    return fractional_lag * sample_interval_s, float(correlations[peak])


def _standardize_axes(values: FloatArray) -> FloatArray:
    centered = values - np.mean(values, axis=0)
    scale = np.std(centered, axis=0)
    return centered / scale


def _correlation_at_shift(
    target: FloatArray,
    comparator: FloatArray,
    shift_samples: int,
) -> float:
    if shift_samples > 0:
        target_part = target[shift_samples:]
        comparator_part = comparator[:-shift_samples]
    elif shift_samples < 0:
        target_part = target[:shift_samples]
        comparator_part = comparator[-shift_samples:]
    else:
        target_part = target
        comparator_part = comparator
    numerator = float(np.sum(target_part * comparator_part))
    denominator = float(np.sqrt(np.sum(target_part**2) * np.sum(comparator_part**2)))
    return numerator / denominator if denominator else float("nan")


def _align_comparator(
    target_rows: FloatArray,
    comparator_rows: FloatArray,
    shift_s: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    elapsed = target_rows[:, 3]
    start = max(float(elapsed[0]), float(elapsed[0] + shift_s))
    end = min(float(elapsed[-1]), float(elapsed[-1] + shift_s))
    mask = (elapsed >= start) & (elapsed <= end)
    common_time = elapsed[mask]
    if common_time.size < 2:
        raise AngleExtremaError(
            "estimated delay leaves fewer than two overlapping rows"
        )
    source_time = common_time - shift_s
    aligned = np.column_stack(
        [
            np.interp(source_time, comparator_rows[:, 3], comparator_rows[:, index])
            for index in range(3)
        ]
    )
    return common_time, target_rows[mask, :3], aligned


def _axis_limits(
    target: FloatArray, comparator: FloatArray
) -> list[tuple[float, float]]:
    limits: list[tuple[float, float]] = []
    for index in range(3):
        low = float(min(np.min(target[:, index]), np.min(comparator[:, index])))
        high = float(max(np.max(target[:, index]), np.max(comparator[:, index])))
        padding = max((high - low) * 0.05, 1e-9)
        limits.append((low - padding, high + padding))
    return limits


def _plot_comparison(
    destination: Path,
    elapsed: FloatArray,
    target: FloatArray,
    comparator: FloatArray,
    *,
    target_label: str,
    comparator_label: str,
    title: str,
    y_limits: list[tuple[float, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns  # type: ignore[import-untyped]

    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    palette = sns.color_palette("colorblind")
    figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    try:
        for index, axis_name in enumerate(_AXES):
            axes[index].plot(
                elapsed,
                target[:, index],
                color=palette[0],
                linewidth=1.8,
                label=target_label,
            )
            axes[index].plot(
                elapsed,
                comparator[:, index],
                color=palette[1],
                linewidth=1.6,
                linestyle=(0, (8, 2)),
                label=comparator_label,
            )
            axes[index].fill_between(
                elapsed,
                target[:, index],
                comparator[:, index],
                color=palette[2],
                alpha=0.12,
                linewidth=0,
            )
            axes[index].set_ylabel(f"{axis_name} angle (deg)")
            axes[index].set_ylim(*y_limits[index])
            axes[index].legend(loc="best")
        axes[-1].set_xlabel("Relative elapsed time (s)")
        figure.suptitle(title)
        figure.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, format="png")
    finally:
        plt.close(figure)


def _source_output_dir(source: Path, base: Path, root: Path) -> Path:
    preprocessing_root = (root / "outputs" / "preprocessing").resolve()
    if source.is_relative_to(preprocessing_root):
        relative = source.relative_to(preprocessing_root)
        return base / "sources" / relative.parent / source.stem
    digest = hashlib.sha256(str(source).encode()).hexdigest()[:8]
    return base / "sources" / "external" / f"{source.stem}-{digest}"


def _normalized_config(config: ComparisonConfig) -> dict[str, object]:
    return {
        "comparison_name": config.comparison_name,
        "analysis_run_name": config.analysis_run_name,
        "target_cutoff_hz": config.target_cutoff_hz,
        "comparison_cutoff_frequencies_hz": list(
            config.comparison_cutoff_frequencies_hz
        ),
        "include_raw": config.include_raw,
        "force": config.force,
    }


def _write_compensation_csv(
    destination: Path,
    records: list[dict[str, object]],
) -> None:
    fields = (
        "source_file",
        "comparator",
        "target",
        "estimated_shift_s",
        "estimated_shift_ms",
        "shift_action",
        "correlation",
        "overlap_start_s",
        "overlap_end_s",
    )
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_readme(destination: Path, config: ComparisonConfig) -> None:
    destination.write_text(
        "# Target integrated angle comparisons\n\n"
        f"Target: {config.target_cutoff_hz:g} Hz fourth-order causal Butterworth "
        "integrated angle.\n\n"
        "`estimated_shift_ms` is one shared tri-axis best-fit translation per "
        "source and comparator. Positive values delay the comparator; negative "
        "values advance it. The estimator maximizes normalized correlation of "
        "standardized first differences. Compensated plots use linear "
        "sub-sample interpolation and omit both non-overlapping edges. This is "
        "an effective alignment for visual comparison, not an exact fixed filter "
        "delay.\n",
        encoding="utf-8",
    )


def _publish(temporary_dir: Path, output_dir: Path, *, force: bool) -> None:
    if not output_dir.exists():
        os.replace(temporary_dir, output_dir)
        return
    if not force:
        raise AngleExtremaError("comparison directory already exists", path=output_dir)
    backup = output_dir.with_name(f".{output_dir.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(output_dir, backup)
    try:
        os.replace(temporary_dir, output_dir)
    except BaseException:
        os.replace(backup, output_dir)
        raise
    shutil.rmtree(backup)


def _write_json(destination: Path, payload: object) -> None:
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _variant_label(variant: str) -> str:
    if variant == "raw":
        return "Raw Integrated Angle"
    cutoff = variant.removeprefix("lowpass_").removesuffix("hz").replace("p", ".")
    return f"{cutoff} Hz causal low-pass"


def _resolve_path(path: Path, root: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def _display_path(path: Path, root: Path) -> str:
    resolved = _resolve_path(path, root)
    return (
        resolved.relative_to(root).as_posix()
        if resolved.is_relative_to(root)
        else resolved.as_posix()
    )


def _clean_float(value: float) -> float:
    return float(f"{value:.12g}")


def _cutoff_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _float_list(value: object, name: str, path: Path) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise AngleExtremaError(f"{name} must be an array of numbers", path=path)
    return tuple(_number(item, name, path) for item in value)


def _number(value: object, name: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AngleExtremaError(f"{name} must be a number", path=path)
    number = float(value)
    if not math.isfinite(number):
        raise AngleExtremaError(f"{name} must be finite", path=path)
    return number


def _boolean(value: object, name: str, path: Path) -> bool:
    if not isinstance(value, bool):
        raise AngleExtremaError(f"{name} must be a boolean", path=path)
    return value
