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
from scipy import signal  # type: ignore[import-untyped]

from imup.angle_extrema.core import AngleExtremaError
from imup.preprocessing import find_repository_root

FloatArray = npt.NDArray[np.float64]

_ANGLE_SUFFIX = "integrated_angle.csv"
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_AXES = ("x", "y", "z")
_ALLOWED_CONFIG_KEYS = frozenset(
    {
        "run_name",
        "input_paths",
        "window_duration_s",
        "stride_duration_s",
        "lower_bound_deg",
        "upper_bound_deg",
        "cutoff_frequencies_hz",
        "violation_plot_input_paths",
        "save_filtered_csv",
        "allow_external_input",
        "force",
    }
)


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for one angle-extrema experiment run."""

    run_name: str
    input_paths: tuple[Path, ...]
    window_duration_s: float = 1.0
    stride_duration_s: float = 0.5
    lower_bound_deg: float = -1.0
    upper_bound_deg: float = 1.0
    cutoff_frequencies_hz: tuple[float, ...] = (11.0, 9.0, 7.0, 5.0, 3.0)
    violation_plot_input_paths: tuple[Path, ...] = ()
    save_filtered_csv: bool = True
    allow_external_input: bool = False
    force: bool = False


@dataclass(frozen=True)
class AnalysisDiagnostic:
    """One path-scoped analysis warning or failure."""

    path: Path | None
    message: str


@dataclass(frozen=True)
class AnalysisRunResult:
    """Observable result of one complete experiment attempt."""

    run_dir: Path
    processed: tuple[Path, ...]
    failures: tuple[AnalysisDiagnostic, ...]
    warnings: tuple[AnalysisDiagnostic, ...]

    @property
    def exit_code(self) -> int:
        return int(bool(self.failures))


@dataclass(frozen=True)
class _AngleData:
    rows: FloatArray
    sample_interval_s: float


def load_analysis_config(
    config_path: Path,
    *,
    repository_root: Path | None = None,
) -> AnalysisConfig:
    """Load and strictly validate an angle-extrema JSON configuration."""
    root = (repository_root or find_repository_root()).resolve()
    resolved_path = _resolve_path(config_path, root)
    try:
        with resolved_path.open(encoding="utf-8") as config_file:
            payload: object = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise AngleExtremaError(
            f"could not read angle-extrema JSON: {error}",
            path=resolved_path,
        ) from error

    if not isinstance(payload, dict):
        raise AngleExtremaError(
            "angle-extrema JSON must contain one object",
            path=resolved_path,
        )
    if not all(isinstance(key, str) for key in payload):
        raise AngleExtremaError(
            "angle-extrema JSON keys must be strings",
            path=resolved_path,
        )
    values = cast(Mapping[str, object], payload)
    unknown = set(values) - _ALLOWED_CONFIG_KEYS
    if unknown:
        raise AngleExtremaError(
            f"unknown angle-extrema JSON keys: {', '.join(sorted(unknown))}",
            path=resolved_path,
        )

    run_name = values.get("run_name")
    if not isinstance(run_name, str) or not run_name:
        raise AngleExtremaError(
            "run_name must be a non-empty string", path=resolved_path
        )

    input_paths = _path_list(values.get("input_paths"), "input_paths", resolved_path)
    violation_paths = _path_list(
        values.get("violation_plot_input_paths", []),
        "violation_plot_input_paths",
        resolved_path,
    )
    cutoffs = _float_list(
        values.get("cutoff_frequencies_hz", [11.0, 9.0, 7.0, 5.0, 3.0]),
        "cutoff_frequencies_hz",
        resolved_path,
    )
    return AnalysisConfig(
        run_name=run_name,
        input_paths=input_paths,
        window_duration_s=_number(
            values.get("window_duration_s", 1.0),
            "window_duration_s",
            resolved_path,
        ),
        stride_duration_s=_number(
            values.get("stride_duration_s", 0.5),
            "stride_duration_s",
            resolved_path,
        ),
        lower_bound_deg=_number(
            values.get("lower_bound_deg", -1.0),
            "lower_bound_deg",
            resolved_path,
        ),
        upper_bound_deg=_number(
            values.get("upper_bound_deg", 1.0),
            "upper_bound_deg",
            resolved_path,
        ),
        cutoff_frequencies_hz=cutoffs,
        violation_plot_input_paths=violation_paths,
        save_filtered_csv=_boolean(
            values.get("save_filtered_csv", True),
            "save_filtered_csv",
            resolved_path,
        ),
        allow_external_input=_boolean(
            values.get("allow_external_input", False),
            "allow_external_input",
            resolved_path,
        ),
        force=_boolean(values.get("force", False), "force", resolved_path),
    )


def run_extrema_analysis(
    config: AnalysisConfig,
    *,
    repository_root: Path | None = None,
) -> AnalysisRunResult:
    """Run one complete angle-extrema experiment and atomically publish it."""
    root = (repository_root or find_repository_root()).resolve()
    _validate_config(config)
    run_dir = root / "outputs" / "angle_extrema" / "runs" / config.run_name
    if run_dir.exists() and not config.force:
        raise AngleExtremaError(
            "run directory already exists; rerun with force=true to replace it",
            path=run_dir,
        )

    input_files, discovery_failures, discovery_warnings = _expand_inputs(config, root)
    plot_inputs = _resolve_plot_inputs(config, root, input_files)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{config.run_name}.", dir=run_dir.parent)
    )
    processed: list[Path] = []
    failures = list(discovery_failures)
    warnings = list(discovery_warnings)
    summaries: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []

    try:
        _write_json(
            temporary_dir / "analysis_config.json",
            _normalized_config(config, root),
        )
        for source in input_files:
            try:
                data = _read_angle_data(source)
                source_output = _source_output_dir(source, temporary_dir, root)
                _plot_raw_series(
                    source_output / "raw_series.png",
                    data,
                    source_name=source.name,
                )
                summary, source_violations = _analyze_variant(
                    source=source,
                    source_label=_display_path(source, root),
                    variant="raw",
                    data=data,
                    config=config,
                )
                summaries.append(summary)
                violations.extend(source_violations)
                if summary["total_windows"] == 0:
                    warnings.append(
                        AnalysisDiagnostic(
                            source,
                            "input is shorter than one complete window",
                        )
                    )
                if source in plot_inputs:
                    _plot_violation_windows(
                        source=source,
                        source_output=source_output,
                        variant="raw",
                        data=data,
                        violations=source_violations,
                        config=config,
                    )
                for cutoff_hz in config.cutoff_frequencies_hz:
                    variant = f"lowpass_{_cutoff_slug(cutoff_hz)}hz"
                    nyquist_hz = 0.5 / data.sample_interval_s
                    if cutoff_hz >= nyquist_hz:
                        failures.append(
                            AnalysisDiagnostic(
                                source,
                                (
                                    f"{variant} cutoff must be below Nyquist "
                                    f"frequency {nyquist_hz:.12g} Hz"
                                ),
                            )
                        )
                        continue
                    filtered = _causal_lowpass(data, cutoff_hz)
                    if config.save_filtered_csv:
                        _write_csv(
                            source_output / "filtered" / f"{variant}.csv",
                            filtered.rows,
                        )
                    filtered_summary, filtered_violations = _analyze_variant(
                        source=source,
                        source_label=_display_path(source, root),
                        variant=variant,
                        data=filtered,
                        config=config,
                    )
                    summaries.append(filtered_summary)
                    violations.extend(filtered_violations)
                    if source in plot_inputs:
                        _plot_violation_windows(
                            source=source,
                            source_output=source_output,
                            variant=variant,
                            data=filtered,
                            violations=filtered_violations,
                            config=config,
                        )
                processed.append(source)
            except (AngleExtremaError, OSError, ValueError) as error:
                failures.append(
                    AnalysisDiagnostic(
                        path=getattr(error, "path", None) or source,
                        message=str(error),
                    )
                )

        results = {
            "run_status": "failed" if failures else "succeeded",
            "summaries": summaries,
            "violations": violations,
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
        }
        _write_json(temporary_dir / "extrema_results.json", results)
        _publish_run(temporary_dir, run_dir, force=config.force)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return AnalysisRunResult(
        run_dir=run_dir,
        processed=tuple(processed),
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


def _validate_config(config: AnalysisConfig) -> None:
    if not _RUN_NAME.fullmatch(config.run_name):
        raise AngleExtremaError(
            "run_name must start with an ASCII letter or number and may then "
            "contain letters, numbers, '.', '_' and '-'"
        )
    if not config.input_paths:
        raise AngleExtremaError("input_paths must contain at least one path")
    for name, value in (
        ("window_duration_s", config.window_duration_s),
        ("stride_duration_s", config.stride_duration_s),
    ):
        if not math.isfinite(value) or value <= 0:
            raise AngleExtremaError(f"{name} must be finite and greater than zero")
    if not all(
        math.isfinite(value)
        for value in (config.lower_bound_deg, config.upper_bound_deg)
    ):
        raise AngleExtremaError("angle bounds must be finite")
    if not config.lower_bound_deg < 0 < config.upper_bound_deg:
        raise AngleExtremaError(
            "angle bounds must satisfy lower_bound_deg < 0 < upper_bound_deg"
        )
    if len(set(config.cutoff_frequencies_hz)) != len(config.cutoff_frequencies_hz):
        raise AngleExtremaError("cutoff_frequencies_hz must not contain duplicates")
    if any(
        not math.isfinite(value) or value <= 0 for value in config.cutoff_frequencies_hz
    ):
        raise AngleExtremaError(
            "cutoff_frequencies_hz values must be finite and greater than zero"
        )
    for name, value in (
        ("save_filtered_csv", config.save_filtered_csv),
        ("allow_external_input", config.allow_external_input),
        ("force", config.force),
    ):
        if not isinstance(value, bool):
            raise AngleExtremaError(f"{name} must be a boolean")


def _expand_inputs(
    config: AnalysisConfig,
    root: Path,
) -> tuple[list[Path], list[AnalysisDiagnostic], list[AnalysisDiagnostic]]:
    preprocessing_root = (root / "outputs" / "preprocessing").resolve()
    expanded: list[Path] = []
    failures: list[AnalysisDiagnostic] = []
    warnings: list[AnalysisDiagnostic] = []
    for configured in config.input_paths:
        resolved = _resolve_path(configured, root)
        if not resolved.exists():
            failures.append(AnalysisDiagnostic(resolved, "input path does not exist"))
            continue
        candidates = (
            [resolved]
            if resolved.is_file()
            else sorted(
                path
                for path in resolved.iterdir()
                if path.is_file()
                and path.name.endswith(_ANGLE_SUFFIX)
                and len(path.name) > len(_ANGLE_SUFFIX)
            )
        )
        if resolved.is_dir() and not candidates:
            failures.append(
                AnalysisDiagnostic(
                    resolved,
                    "directory contains no direct *integrated_angle.csv files",
                )
            )
        for candidate in candidates:
            if not candidate.name.endswith(_ANGLE_SUFFIX):
                failures.append(
                    AnalysisDiagnostic(
                        candidate,
                        "input file must end with integrated_angle.csv",
                    )
                )
                continue
            if (
                not candidate.is_relative_to(preprocessing_root)
                and not config.allow_external_input
            ):
                message = (
                    "input is outside outputs/preprocessing; "
                    "set allow_external_input=true to process it"
                )
                warnings.append(AnalysisDiagnostic(candidate, message))
                failures.append(AnalysisDiagnostic(candidate, message))
                continue
            expanded.append(candidate)

    duplicates = sorted(
        {path for path in expanded if expanded.count(path) > 1},
        key=lambda item: item.as_posix(),
    )
    if duplicates:
        rendered = ", ".join(_display_path(path, root) for path in duplicates)
        raise AngleExtremaError(f"input_paths resolve to duplicate files: {rendered}")
    return expanded, failures, warnings


def _resolve_plot_inputs(
    config: AnalysisConfig,
    root: Path,
    input_files: list[Path],
) -> set[Path]:
    requested = {
        _resolve_path(path, root) for path in config.violation_plot_input_paths
    }
    unknown = sorted(requested - set(input_files), key=lambda item: item.as_posix())
    if unknown:
        rendered = ", ".join(_display_path(path, root) for path in unknown)
        raise AngleExtremaError(
            f"violation_plot_input_paths are not analyzed input files: {rendered}"
        )
    return requested


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
                        f"line {line_number} contains a non-numeric value",
                        path=source,
                    ) from error
                if not all(math.isfinite(value) for value in parsed):
                    raise AngleExtremaError(
                        f"line {line_number} contains a non-finite value",
                        path=source,
                    )
                rows.append(parsed)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise AngleExtremaError(
            f"could not read angle CSV: {error}", path=source
        ) from error

    if len(rows) < 2:
        raise AngleExtremaError("at least two angle rows are required", path=source)
    values = np.asarray(rows, dtype=np.float64)
    elapsed = values[:, 3]
    if not math.isclose(float(elapsed[0]), 0.0, abs_tol=1e-12):
        raise AngleExtremaError("elapsed_s must start at zero", path=source)
    deltas = np.diff(elapsed)
    if np.any(deltas <= 0):
        raise AngleExtremaError("elapsed_s must be strictly increasing", path=source)
    sample_interval = float(np.median(deltas))
    if not np.allclose(deltas, sample_interval, rtol=1e-6, atol=1e-12):
        raise AngleExtremaError(
            "elapsed_s intervals must be fixed within floating-point tolerance",
            path=source,
        )
    return _AngleData(rows=values, sample_interval_s=sample_interval)


def _analyze_variant(
    *,
    source: Path,
    source_label: str,
    variant: str,
    data: _AngleData,
    config: AnalysisConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    del source
    width = math.ceil(config.window_duration_s / data.sample_interval_s)
    stride = math.ceil(config.stride_duration_s / data.sample_interval_s)
    starts = range(0, max(0, data.rows.shape[0] - width + 1), stride)
    records: list[dict[str, object]] = []
    total = 0
    for window_index, start in enumerate(starts):
        total += 1
        window = data.rows[start : start + width, :3]
        relative = window - window[0]
        minima = np.min(relative, axis=0)
        maxima = np.max(relative, axis=0)
        violation_axes: dict[str, list[str]] = {}
        extrema = {
            axis: {
                "min_deg": _clean_float(float(minima[index])),
                "max_deg": _clean_float(float(maxima[index])),
            }
            for index, axis in enumerate(_AXES)
        }
        for index, axis in enumerate(_AXES):
            directions: list[str] = []
            if minima[index] < config.lower_bound_deg:
                directions.append("below_lower_bound")
            if maxima[index] > config.upper_bound_deg:
                directions.append("above_upper_bound")
            if directions:
                violation_axes[axis] = directions
        if not violation_axes:
            continue
        start_elapsed = float(data.rows[start, 3])
        records.append(
            {
                "source_file": source_label,
                "signal_variant": variant,
                "window_index": window_index,
                "start_row": start,
                "start_elapsed_s": _clean_float(start_elapsed),
                "end_elapsed_s_exclusive": _clean_float(
                    start_elapsed + width * data.sample_interval_s
                ),
                "axis_extrema": extrema,
                "violations": violation_axes,
            }
        )
    return (
        {
            "source_file": source_label,
            "signal_variant": variant,
            "sample_interval_s": _clean_float(data.sample_interval_s),
            "estimated_frequency_hz": _clean_float(1.0 / data.sample_interval_s),
            "total_windows": total,
            "violating_windows": len(records),
        },
        records,
    )


def _causal_lowpass(data: _AngleData, cutoff_hz: float) -> _AngleData:
    sampling_frequency_hz = 1.0 / data.sample_interval_s
    sections = signal.butter(
        4,
        cutoff_hz,
        btype="lowpass",
        fs=sampling_frequency_hz,
        output="sos",
    )
    steady_state = signal.sosfilt_zi(sections)
    filtered_rows = data.rows.copy()
    for index in range(3):
        filtered_rows[:, index], _ = signal.sosfilt(
            sections,
            data.rows[:, index],
            zi=steady_state * data.rows[0, index],
        )
    return _AngleData(
        rows=filtered_rows,
        sample_interval_s=data.sample_interval_s,
    )


def _plot_raw_series(
    destination: Path,
    data: _AngleData,
    *,
    source_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns  # type: ignore[import-untyped]

    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    palette = sns.color_palette("colorblind", 3)
    figure, axes = plt.subplots(figsize=(12, 6))
    try:
        for index, axis in enumerate(_AXES):
            sns.lineplot(
                x=data.rows[:, 3],
                y=data.rows[:, index],
                estimator=None,
                errorbar=None,
                label=axis,
                color=palette[index],
                ax=axes,
            )
        axes.set_xlabel("Relative elapsed time (s)")
        axes.set_ylabel("Integrated angle (deg)")
        axes.set_title(f"{source_name} | raw integrated angle")
        axes.legend(title="Axis")
        figure.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, format="png")
    finally:
        plt.close(figure)


def _plot_violation_windows(
    *,
    source: Path,
    source_output: Path,
    variant: str,
    data: _AngleData,
    violations: list[dict[str, object]],
    config: AnalysisConfig,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    palette = sns.color_palette("colorblind", 3)
    width = math.ceil(config.window_duration_s / data.sample_interval_s)
    destination_dir = source_output / "violation_windows" / variant
    for record in violations:
        start = cast(int, record["start_row"])
        window_index = cast(int, record["window_index"])
        start_elapsed = cast(float, record["start_elapsed_s"])
        window = data.rows[start : start + width]
        relative_angles = window[:, :3] - window[0, :3]
        relative_time = window[:, 3] - window[0, 3]

        figure, axes = plt.subplots(figsize=(10, 5))
        try:
            for index, axis in enumerate(_AXES):
                sns.lineplot(
                    x=relative_time,
                    y=relative_angles[:, index],
                    estimator=None,
                    errorbar=None,
                    label=axis,
                    color=palette[index],
                    ax=axes,
                )
            axes.axhline(
                config.upper_bound_deg,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="upper bound",
            )
            axes.axhline(
                config.lower_bound_deg,
                color="black",
                linestyle=":",
                linewidth=1.2,
                label="lower bound",
            )
            axes.set_xlabel("Window-relative time (s)")
            axes.set_ylabel("Window-relative angle (deg)")
            axes.set_title(f"{source.name} | {variant} | start={start_elapsed:g}s")
            axes.legend(title="Axis / bound")
            figure.tight_layout()
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / (
                f"window_{window_index:06d}_start_{_time_slug(start_elapsed)}s.png"
            )
            figure.savefig(destination, dpi=150, format="png")
        finally:
            plt.close(figure)


def _source_output_dir(source: Path, temporary_dir: Path, root: Path) -> Path:
    preprocessing_root = (root / "outputs" / "preprocessing").resolve()
    if source.is_relative_to(preprocessing_root):
        relative = source.relative_to(preprocessing_root)
        return temporary_dir / "sources" / relative.parent / source.stem
    digest = hashlib.sha256(str(source).encode()).hexdigest()[:8]
    return temporary_dir / "sources" / "external" / f"{source.stem}-{digest}"


def _normalized_config(config: AnalysisConfig, root: Path) -> dict[str, object]:
    return {
        "run_name": config.run_name,
        "input_paths": [_display_path(path, root) for path in config.input_paths],
        "window_duration_s": config.window_duration_s,
        "stride_duration_s": config.stride_duration_s,
        "lower_bound_deg": config.lower_bound_deg,
        "upper_bound_deg": config.upper_bound_deg,
        "cutoff_frequencies_hz": list(config.cutoff_frequencies_hz),
        "violation_plot_input_paths": [
            _display_path(path, root) for path in config.violation_plot_input_paths
        ],
        "save_filtered_csv": config.save_filtered_csv,
        "allow_external_input": config.allow_external_input,
        "force": config.force,
    }


def _publish_run(temporary_dir: Path, run_dir: Path, *, force: bool) -> None:
    if not run_dir.exists():
        os.replace(temporary_dir, run_dir)
        return
    if not force:
        raise AngleExtremaError("run directory already exists", path=run_dir)
    backup = run_dir.with_name(f".{run_dir.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(run_dir, backup)
    try:
        os.replace(temporary_dir, run_dir)
    except BaseException:
        os.replace(backup, run_dir)
        raise
    shutil.rmtree(backup)


def _write_json(destination: Path, payload: object) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(destination: Path, rows: FloatArray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows.tolist())


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


def _time_slug(value: float) -> str:
    return f"{value:.12g}".replace(".", "p")


def _path_list(value: object, name: str, path: Path) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        if name == "violation_plot_input_paths" and value == []:
            return ()
        raise AngleExtremaError(
            f"{name} must be a non-empty array of strings", path=path
        )
    if not all(isinstance(item, str) and item for item in value):
        raise AngleExtremaError(
            f"{name} must contain only non-empty strings", path=path
        )
    return tuple(Path(cast(str, item)) for item in value)


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
