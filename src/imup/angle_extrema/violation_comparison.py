from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from imup.angle_extrema.comparison import (
    _cutoff_slug,
    _display_path,
    _read_angle_data,
    _require_shared_time_axis,
    _source_output_dir,
    _variant_label,
    load_comparison_config,
)
from imup.angle_extrema.core import AngleExtremaError
from imup.preprocessing import find_repository_root

FloatArray = npt.NDArray[np.float64]

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_AXES = ("x", "y", "z")
_ALLOWED_CONFIG_KEYS = frozenset({"comparison_name", "analysis_run_name", "force"})
_TARGET_COLOR = "#0072B2"
_COMPARATOR_COLORS = {
    "raw": "#4D4D4D",
    "lowpass_3hz": "#CC79A7",
    "lowpass_5hz": "#E69F00",
    "lowpass_7hz": "#009E73",
    "lowpass_9hz": "#D55E00",
}


@dataclass(frozen=True)
class ViolationComparisonConfig:
    """Configuration for delay-compensated target-violation comparisons."""

    comparison_name: str
    analysis_run_name: str
    force: bool = False


@dataclass(frozen=True)
class ViolationComparisonDiagnostic:
    """One path-scoped target-violation comparison warning or failure."""

    path: Path | None
    message: str


@dataclass(frozen=True)
class ViolationComparisonRunResult:
    """Observable result of one target-violation comparison attempt."""

    output_dir: Path
    processed_sources: int
    target_windows: int
    plots: int
    failures: tuple[ViolationComparisonDiagnostic, ...]
    warnings: tuple[ViolationComparisonDiagnostic, ...]

    @property
    def exit_code(self) -> int:
        return int(bool(self.failures))


@dataclass(frozen=True)
class _TargetViolation:
    source_file: str
    window_index: int
    start_row: int
    start_elapsed_s: float
    end_elapsed_s_exclusive: float
    violated_axes: tuple[str, ...]


@dataclass(frozen=True)
class _Shift:
    source_file: str
    comparator: str
    target: str
    shift_s: float
    shift_ms: float
    action: str
    correlation: float


@dataclass(frozen=True)
class _PreparedWindow:
    target_time_s: FloatArray
    target_angles_deg: FloatArray
    comparator_time_s: FloatArray
    comparator_angles_deg: FloatArray
    requested_query_start_s: float
    requested_query_end_s_exclusive: float
    plotted_query_start_s: float
    plotted_query_end_s: float
    coverage_status: str
    coverage_fraction: float


@dataclass(frozen=True)
class _RenderJob:
    root: Path
    temporary_dir: Path
    output_dir: Path
    source: Path
    target_rows: FloatArray
    comparator_rows: FloatArray
    violations: tuple[_TargetViolation, ...]
    comparator_variant: str
    target_variant: str
    shift: _Shift
    lower_bound_deg: float
    upper_bound_deg: float


@dataclass(frozen=True)
class _RenderResult:
    source_file: str
    records: tuple[dict[str, object], ...]
    failures: tuple[ViolationComparisonDiagnostic, ...]
    warnings: tuple[ViolationComparisonDiagnostic, ...]


def load_violation_comparison_config(
    config_path: Path,
    *,
    repository_root: Path | None = None,
) -> ViolationComparisonConfig:
    """Load and strictly validate one violation-comparison JSON object."""
    root = (repository_root or find_repository_root()).resolve()
    resolved_path = _resolve_path(config_path, root)
    values = _read_json_object(resolved_path, "violation comparison JSON")
    unknown = set(values) - _ALLOWED_CONFIG_KEYS
    if unknown:
        raise AngleExtremaError(
            f"unknown violation comparison JSON keys: {', '.join(sorted(unknown))}",
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
    force = values.get("force", False)
    if not isinstance(force, bool):
        raise AngleExtremaError("force must be a boolean", path=resolved_path)
    config = ViolationComparisonConfig(
        comparison_name=comparison_name,
        analysis_run_name=analysis_run_name,
        force=force,
    )
    _validate_config(config)
    return config


def run_violation_comparison(
    config: ViolationComparisonConfig,
    *,
    repository_root: Path | None = None,
) -> ViolationComparisonRunResult:
    """Render pairwise comparisons for every Target Violation Window."""
    root = (repository_root or find_repository_root()).resolve()
    _validate_config(config)
    comparison_dir = (
        root / "outputs" / "angle_extrema" / "comparisons" / config.comparison_name
    )
    analysis_run = (
        root / "outputs" / "angle_extrema" / "runs" / config.analysis_run_name
    )
    output_dir = comparison_dir / "target_violation_windows"
    if not comparison_dir.is_dir():
        raise AngleExtremaError(
            "comparison directory does not exist", path=comparison_dir
        )
    if not analysis_run.is_dir():
        raise AngleExtremaError(
            "analysis run directory does not exist", path=analysis_run
        )
    if output_dir.exists() and not config.force:
        raise AngleExtremaError(
            "target violation comparison output already exists; rerun with "
            "force=true to replace it",
            path=output_dir,
        )

    comparison_config = load_comparison_config(
        comparison_dir / "comparison_config.json", repository_root=root
    )
    if comparison_config.comparison_name != config.comparison_name:
        raise AngleExtremaError(
            "saved comparison name does not match the requested comparison",
            path=comparison_dir / "comparison_config.json",
        )
    if comparison_config.analysis_run_name != config.analysis_run_name:
        raise AngleExtremaError(
            "saved comparison uses a different analysis run",
            path=comparison_dir / "comparison_config.json",
        )
    if comparison_config.target_cutoff_hz != 11.0:
        raise AngleExtremaError(
            "target violation comparisons require the 11 Hz Target Integrated Angle",
            path=comparison_dir / "comparison_config.json",
        )

    target_variant = f"lowpass_{_cutoff_slug(comparison_config.target_cutoff_hz)}hz"
    expected_comparators = [
        *(["raw"] if comparison_config.include_raw else []),
        *[
            f"lowpass_{_cutoff_slug(cutoff)}hz"
            for cutoff in comparison_config.comparison_cutoff_frequencies_hz
        ],
    ]
    analysis_config_path = analysis_run / "analysis_config.json"
    analysis_config = _read_json_object(analysis_config_path, "analysis configuration")
    lower_bound = _number(analysis_config, "lower_bound_deg", analysis_config_path)
    upper_bound = _number(analysis_config, "upper_bound_deg", analysis_config_path)
    if lower_bound >= upper_bound:
        raise AngleExtremaError(
            "analysis lower_bound_deg must be less than upper_bound_deg",
            path=analysis_config_path,
        )
    violations = _read_target_violations(
        analysis_run / "extrema_results.json", target_variant
    )
    shifts = _read_shifts(comparison_dir / "delay_compensation.csv", target_variant)
    shift_by_pair = {(shift.source_file, shift.comparator): shift for shift in shifts}
    violations_by_source: dict[str, list[_TargetViolation]] = defaultdict(list)
    for violation in violations:
        violations_by_source[violation.source_file].append(violation)

    temporary_dir = Path(
        tempfile.mkdtemp(prefix=".target_violation_windows.", dir=comparison_dir)
    )
    records: list[dict[str, object]] = []
    failures: list[ViolationComparisonDiagnostic] = []
    warnings: list[ViolationComparisonDiagnostic] = []
    processed_source_files: set[str] = set()
    try:
        _write_json(
            temporary_dir / "violation_comparison_config.json",
            {
                "comparison_name": config.comparison_name,
                "analysis_run_name": config.analysis_run_name,
                "force": config.force,
            },
        )
        jobs: list[_RenderJob] = []
        for source_file, source_violations in violations_by_source.items():
            source = _resolve_path(Path(source_file), root)
            try:
                raw = _read_angle_data(source)
                source_run_dir = _source_output_dir(source, analysis_run, root)
                target_path = source_run_dir / "filtered" / f"{target_variant}.csv"
                target = _read_angle_data(target_path)
                _require_shared_time_axis(raw, target, source, target_path)
            except (AngleExtremaError, OSError, ValueError) as error:
                failures.append(
                    ViolationComparisonDiagnostic(
                        path=getattr(error, "path", None) or source,
                        message=str(error),
                    )
                )
                continue

            for comparator_variant in expected_comparators:
                shift = shift_by_pair.get((source_file, comparator_variant))
                if shift is None:
                    failures.append(
                        ViolationComparisonDiagnostic(
                            comparison_dir / "delay_compensation.csv",
                            f"missing {comparator_variant} delay for {source_file}",
                        )
                    )
                    continue
                comparator_path = (
                    source
                    if comparator_variant == "raw"
                    else source_run_dir / "filtered" / f"{comparator_variant}.csv"
                )
                try:
                    comparator = _read_angle_data(comparator_path)
                    _require_shared_time_axis(
                        target, comparator, target_path, comparator_path
                    )
                except (AngleExtremaError, OSError, ValueError) as error:
                    failures.append(
                        ViolationComparisonDiagnostic(
                            path=getattr(error, "path", None) or comparator_path,
                            message=str(error),
                        )
                    )
                    continue

                jobs.append(
                    _RenderJob(
                        root=root,
                        temporary_dir=temporary_dir,
                        output_dir=output_dir,
                        source=source,
                        target_rows=target.rows,
                        comparator_rows=comparator.rows,
                        violations=tuple(source_violations),
                        comparator_variant=comparator_variant,
                        target_variant=target_variant,
                        shift=shift,
                        lower_bound_deg=lower_bound,
                        upper_bound_deg=upper_bound,
                    )
                )

        if not jobs:
            render_results = []
        elif len(jobs) == 1:
            render_results = [_render_job(jobs[0])]
        else:
            worker_count = min(8, len(jobs), os.cpu_count() or 1)
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                render_results = list(executor.map(_render_job, jobs, chunksize=1))
        for render_result in render_results:
            records.extend(render_result.records)
            failures.extend(render_result.failures)
            warnings.extend(render_result.warnings)
            if render_result.records:
                processed_source_files.add(render_result.source_file)

        processed_sources = len(processed_source_files)

        _write_window_csv(temporary_dir / "window_comparisons.csv", records)
        _write_readme(
            temporary_dir / "README.md",
            target_variant=target_variant,
            lower_bound_deg=lower_bound,
            upper_bound_deg=upper_bound,
        )
        partial_count = sum(
            record["coverage_status"] == "partial" for record in records
        )
        _write_json(
            temporary_dir / "violation_comparison_results.json",
            {
                "status": "failed" if failures else "succeeded",
                "processed_sources": processed_sources,
                "target_violation_windows": len(violations),
                "plots": len(records),
                "partial_coverages": partial_count,
                "failures": [_diagnostic_payload(item, root) for item in failures],
                "warnings": [_diagnostic_payload(item, root) for item in warnings],
            },
        )
        _publish(temporary_dir, output_dir, force=config.force)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return ViolationComparisonRunResult(
        output_dir=output_dir,
        processed_sources=processed_sources,
        target_windows=len(violations),
        plots=len(records),
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


def _render_job(job: _RenderJob) -> _RenderResult:
    records: list[dict[str, object]] = []
    failures: list[ViolationComparisonDiagnostic] = []
    warnings: list[ViolationComparisonDiagnostic] = []
    comparator_output = (
        _source_output_dir(job.source, job.temporary_dir, job.root)
        / job.comparator_variant
    )
    for violation in job.violations:
        filename = (
            f"window_{violation.window_index:06d}_start_"
            f"{_time_slug(violation.start_elapsed_s)}s.png"
        )
        plot_path = comparator_output / filename
        final_plot_path = job.output_dir / plot_path.relative_to(job.temporary_dir)
        try:
            prepared = _prepare_window(
                job.target_rows,
                job.comparator_rows,
                start_row=violation.start_row,
                end_elapsed_s_exclusive=violation.end_elapsed_s_exclusive,
                shift_s=job.shift.shift_s,
            )
            _plot_window_comparison(
                plot_path,
                prepared,
                comparator_variant=job.comparator_variant,
                violated_axes=violation.violated_axes,
                lower_bound_deg=job.lower_bound_deg,
                upper_bound_deg=job.upper_bound_deg,
                window_duration_s=(
                    violation.end_elapsed_s_exclusive - violation.start_elapsed_s
                ),
            )
        except (AngleExtremaError, OSError, ValueError) as error:
            failures.append(
                ViolationComparisonDiagnostic(
                    path=final_plot_path,
                    message=str(error),
                )
            )
            continue

        if prepared.coverage_status == "partial":
            warnings.append(
                ViolationComparisonDiagnostic(
                    path=final_plot_path,
                    message=(
                        "PARTIAL_COVERAGE: delay compensation leaves part of the "
                        "target window without comparator data"
                    ),
                )
            )
        records.append(
            _window_record(
                root=job.root,
                output_path=final_plot_path,
                violation=violation,
                comparator_variant=job.comparator_variant,
                target_variant=job.target_variant,
                shift=job.shift,
                prepared=prepared,
            )
        )
    return _RenderResult(
        source_file=job.violations[0].source_file,
        records=tuple(records),
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


def _prepare_window(
    target_rows: FloatArray,
    comparator_rows: FloatArray,
    *,
    start_row: int,
    end_elapsed_s_exclusive: float,
    shift_s: float,
) -> _PreparedWindow:
    """Extract, align and independently rebase one target/comparator window."""
    if start_row < 0 or start_row >= target_rows.shape[0]:
        raise AngleExtremaError("target violation start_row is outside the signal")
    start_elapsed_s = float(target_rows[start_row, 3])
    end_row = int(
        np.searchsorted(target_rows[:, 3], end_elapsed_s_exclusive, side="left")
    )
    if end_row - start_row < 2:
        raise AngleExtremaError("target violation window has fewer than two rows")
    target_window = target_rows[start_row:end_row]
    target_time = target_window[:, 3] - start_elapsed_s
    target_angles = target_window[:, :3] - target_window[0, :3]
    query_time = target_window[:, 3] - shift_s
    valid = (query_time >= comparator_rows[0, 3]) & (
        query_time <= comparator_rows[-1, 3]
    )
    if np.count_nonzero(valid) < 2:
        raise AngleExtremaError(
            "delay compensation leaves fewer than two comparator rows in the window"
        )
    valid_query = query_time[valid]
    comparator_angles = np.column_stack(
        [
            np.interp(
                valid_query,
                comparator_rows[:, 3],
                comparator_rows[:, axis_index],
            )
            for axis_index in range(3)
        ]
    )
    comparator_angles -= comparator_angles[0]
    return _PreparedWindow(
        target_time_s=target_time,
        target_angles_deg=target_angles,
        comparator_time_s=target_time[valid],
        comparator_angles_deg=comparator_angles,
        requested_query_start_s=start_elapsed_s - shift_s,
        requested_query_end_s_exclusive=end_elapsed_s_exclusive - shift_s,
        plotted_query_start_s=float(valid_query[0]),
        plotted_query_end_s=float(valid_query[-1]),
        coverage_status="full" if bool(np.all(valid)) else "partial",
        coverage_fraction=float(np.count_nonzero(valid) / valid.size),
    )


def _plot_window_comparison(
    destination: Path,
    prepared: _PreparedWindow,
    *,
    comparator_variant: str,
    violated_axes: tuple[str, ...],
    lower_bound_deg: float,
    upper_bound_deg: float,
    window_duration_s: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparator_color = _COMPARATOR_COLORS.get(comparator_variant, "#D55E00")
    figure, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    try:
        for index, axis_name in enumerate(_AXES):
            target_values = prepared.target_angles_deg[:, index]
            comparator_values = prepared.comparator_angles_deg[:, index]
            target_for_fill = np.interp(
                prepared.comparator_time_s,
                prepared.target_time_s,
                target_values,
            )
            axes[index].plot(
                prepared.target_time_s,
                target_values,
                color=_TARGET_COLOR,
                linewidth=2.2,
                linestyle="-",
                label="Target Integrated Angle (11 Hz)",
            )
            axes[index].plot(
                prepared.comparator_time_s,
                comparator_values,
                color=comparator_color,
                linewidth=1.7,
                linestyle="-",
                label=_variant_label(comparator_variant),
            )
            axes[index].fill_between(
                prepared.comparator_time_s,
                target_for_fill,
                comparator_values,
                color=comparator_color,
                alpha=0.14,
                linewidth=0,
            )
            axes[index].axhline(
                upper_bound_deg,
                color="#666666",
                linewidth=1.1,
                linestyle=(0, (2, 2)),
                label=f"upper bound ({upper_bound_deg:g}°)",
            )
            axes[index].axhline(
                lower_bound_deg,
                color="#999999",
                linewidth=1.1,
                linestyle=(0, (2, 2)),
                label=f"lower bound ({lower_bound_deg:g}°)",
            )
            all_values = np.concatenate(
                [
                    target_values,
                    comparator_values,
                    np.asarray([lower_bound_deg, upper_bound_deg]),
                ]
            )
            low = float(np.min(all_values))
            high = float(np.max(all_values))
            padding = max((high - low) * 0.08, 0.05)
            axes[index].set_ylim(low - padding, high + padding)
            axes[index].set_xlim(0.0, window_duration_s)
            axes[index].set_ylabel(f"{axis_name} angle (deg)")
            marker = " — VIOLATION" if axis_name in violated_axes else ""
            axes[index].set_title(f"{axis_name.upper()} axis{marker}", loc="left")
            axes[index].grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
            axes[index].legend(loc="best", fontsize="small")
        axes[-1].set_xlabel("Time within target window (s)")
        figure.suptitle("Target Violation Window — With Estimated Delay Compensation")
        figure.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, format="png")
    finally:
        plt.close(figure)


def _read_target_violations(
    results_path: Path,
    target_variant: str,
) -> list[_TargetViolation]:
    payload = _read_json_object(results_path, "angle-extrema results")
    raw_violations = payload.get("violations")
    if not isinstance(raw_violations, list):
        raise AngleExtremaError(
            "angle-extrema results must contain a violations array",
            path=results_path,
        )
    violations: list[_TargetViolation] = []
    for raw_record in raw_violations:
        if not isinstance(raw_record, dict):
            raise AngleExtremaError(
                "angle-extrema violation records must be objects", path=results_path
            )
        record = cast(Mapping[str, object], raw_record)
        if record.get("signal_variant") != target_variant:
            continue
        source_file = record.get("source_file")
        window_index = record.get("window_index")
        start_row = record.get("start_row")
        start_elapsed = record.get("start_elapsed_s")
        end_elapsed = record.get("end_elapsed_s_exclusive")
        raw_axes = record.get("violations")
        if not isinstance(source_file, str) or not source_file:
            raise AngleExtremaError(
                "target violation source_file must be a non-empty string",
                path=results_path,
            )
        if not isinstance(window_index, int) or isinstance(window_index, bool):
            raise AngleExtremaError(
                "target violation window_index must be an integer", path=results_path
            )
        if not isinstance(start_row, int) or isinstance(start_row, bool):
            raise AngleExtremaError(
                "target violation start_row must be an integer", path=results_path
            )
        if not _is_number(start_elapsed) or not _is_number(end_elapsed):
            raise AngleExtremaError(
                "target violation elapsed times must be numbers", path=results_path
            )
        if not isinstance(raw_axes, dict) or not raw_axes:
            raise AngleExtremaError(
                "target violation must name at least one violated axis",
                path=results_path,
            )
        axes = tuple(axis for axis in _AXES if axis in raw_axes)
        if not axes or set(raw_axes) - set(_AXES):
            raise AngleExtremaError(
                "target violation contains an unknown axis", path=results_path
            )
        violations.append(
            _TargetViolation(
                source_file=source_file,
                window_index=window_index,
                start_row=start_row,
                start_elapsed_s=float(cast(float, start_elapsed)),
                end_elapsed_s_exclusive=float(cast(float, end_elapsed)),
                violated_axes=axes,
            )
        )
    return violations


def _read_shifts(path: Path, target_variant: str) -> list[_Shift]:
    shifts: list[_Shift] = []
    try:
        with path.open(encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required = {
                "source_file",
                "comparator",
                "target",
                "estimated_shift_s",
                "estimated_shift_ms",
                "shift_action",
                "correlation",
            }
            if reader.fieldnames is None or required - set(reader.fieldnames):
                raise AngleExtremaError(
                    "delay compensation CSV is missing required columns", path=path
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    shift_s = float(row["estimated_shift_s"])
                    shift_ms = float(row["estimated_shift_ms"])
                    correlation = float(row["correlation"])
                except (TypeError, ValueError) as error:
                    raise AngleExtremaError(
                        f"delay compensation line {line_number} has invalid numbers",
                        path=path,
                    ) from error
                source_file = row["source_file"]
                comparator = row["comparator"]
                target = row["target"]
                action = row["shift_action"]
                if (
                    not source_file
                    or not comparator
                    or target != target_variant
                    or action not in {"delay", "advance", "none"}
                    or not all(
                        math.isfinite(value)
                        for value in (shift_s, shift_ms, correlation)
                    )
                ):
                    raise AngleExtremaError(
                        f"delay compensation line {line_number} is invalid", path=path
                    )
                shifts.append(
                    _Shift(
                        source_file=source_file,
                        comparator=comparator,
                        target=target,
                        shift_s=shift_s,
                        shift_ms=shift_ms,
                        action=action,
                        correlation=correlation,
                    )
                )
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise AngleExtremaError(
            f"could not read delay compensation CSV: {error}", path=path
        ) from error
    return shifts


def _window_record(
    *,
    root: Path,
    output_path: Path,
    violation: _TargetViolation,
    comparator_variant: str,
    target_variant: str,
    shift: _Shift,
    prepared: _PreparedWindow,
) -> dict[str, object]:
    return {
        "source_file": violation.source_file,
        "target": target_variant,
        "comparator": comparator_variant,
        "target_window_index": violation.window_index,
        "target_start_elapsed_s": _clean_float(violation.start_elapsed_s),
        "target_end_elapsed_s_exclusive": _clean_float(
            violation.end_elapsed_s_exclusive
        ),
        "violated_axes": ",".join(violation.violated_axes),
        "estimated_shift_s": _clean_float(shift.shift_s),
        "estimated_shift_ms": _clean_float(shift.shift_ms),
        "shift_action": shift.action,
        "correlation": _clean_float(shift.correlation),
        "requested_query_start_s": _clean_float(prepared.requested_query_start_s),
        "requested_query_end_s_exclusive": _clean_float(
            prepared.requested_query_end_s_exclusive
        ),
        "plotted_query_start_s": _clean_float(prepared.plotted_query_start_s),
        "plotted_query_end_s": _clean_float(prepared.plotted_query_end_s),
        "coverage_status": prepared.coverage_status,
        "coverage_fraction": _clean_float(prepared.coverage_fraction),
        "plot_path": _display_path(output_path, root),
    }


def _write_window_csv(
    destination: Path,
    records: list[dict[str, object]],
) -> None:
    fields = (
        "source_file",
        "target",
        "comparator",
        "target_window_index",
        "target_start_elapsed_s",
        "target_end_elapsed_s_exclusive",
        "violated_axes",
        "estimated_shift_s",
        "estimated_shift_ms",
        "shift_action",
        "correlation",
        "requested_query_start_s",
        "requested_query_end_s_exclusive",
        "plotted_query_start_s",
        "plotted_query_end_s",
        "coverage_status",
        "coverage_fraction",
        "plot_path",
    )
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_readme(
    destination: Path,
    *,
    target_variant: str,
    lower_bound_deg: float,
    upper_bound_deg: float,
) -> None:
    destination.write_text(
        "# Target Violation Window comparisons\n\n"
        f"Windows are selected only from `{target_variant}` and use strict "
        f"bounds `{lower_bound_deg:g}°` and `{upper_bound_deg:g}°`. Each "
        "plot compares the independently window-rebased target with one "
        "delay-compensated diagnostic variant. Both curves are solid and start "
        "at zero whenever the comparator covers the complete target window.\n\n"
        "`window_comparisons.csv` records the shared full-series tri-axis shift, "
        "the signed comparator query interval, violated axes, coverage, and plot "
        "path. Partial boundary coverage is never extrapolated or wrapped; the "
        "comparator begins at its first available point and is rebased there.\n",
        encoding="utf-8",
    )


def _validate_config(config: ViolationComparisonConfig) -> None:
    for name, value in (
        ("comparison_name", config.comparison_name),
        ("analysis_run_name", config.analysis_run_name),
    ):
        if not _NAME.fullmatch(value):
            raise AngleExtremaError(
                f"{name} must start with an ASCII letter or number and may then "
                "contain letters, numbers, '.', '_' and '-'"
            )
    if not isinstance(config.force, bool):
        raise AngleExtremaError("force must be a boolean")


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AngleExtremaError(
            f"could not read {description}: {error}", path=path
        ) from error
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise AngleExtremaError(f"{description} must contain one object", path=path)
    return cast(dict[str, object], payload)


def _number(values: Mapping[str, object], name: str, path: Path) -> float:
    value = values.get(name)
    if not _is_number(value) or not math.isfinite(float(cast(float, value))):
        raise AngleExtremaError(f"{name} must be a finite number", path=path)
    return float(cast(float, value))


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _diagnostic_payload(
    diagnostic: ViolationComparisonDiagnostic, root: Path
) -> dict[str, object]:
    return {
        "path": (
            _display_path(diagnostic.path, root)
            if diagnostic.path is not None
            else None
        ),
        "message": diagnostic.message,
    }


def _publish(temporary_dir: Path, output_dir: Path, *, force: bool) -> None:
    if not output_dir.exists():
        os.replace(temporary_dir, output_dir)
        return
    if not force:
        raise AngleExtremaError(
            "target violation comparison output already exists", path=output_dir
        )
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


def _resolve_path(path: Path, root: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def _clean_float(value: float) -> float:
    return float(f"{value:.12g}")


def _time_slug(value: float) -> str:
    return f"{value:.12g}".replace(".", "p")
