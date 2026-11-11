from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_GYRO_SUFFIX = "gyro.csv"
_ACCEL_SUFFIX = "accel.csv"
_CONFIG_FILENAME = "preprocessing_config.json"
_ALLOWED_CONFIG_KEYS = frozenset({"input_dir", "force"})


class PreprocessingError(Exception):
    """Base error for invalid preprocessing requests."""

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        super().__init__(message)
        self.path = path


class _DataValidationError(PreprocessingError):
    """Raised when one gyroscope CSV violates the input contract."""


@dataclass(frozen=True)
class PreprocessConfig:
    """User-configurable preprocessing parameters."""

    input_dir: Path
    force: bool = False


@dataclass(frozen=True)
class FileResult:
    """Measurements and paths for one successfully processed gyro file."""

    source: Path
    destination: Path
    output_rows: int
    sample_interval_seconds: float
    estimated_frequency_hz: float
    fraction_over_ten_percent: float
    max_interval_ratio: float


@dataclass(frozen=True)
class Diagnostic:
    """A path-scoped warning or error returned by the library API."""

    path: Path | None
    message: str


@dataclass(frozen=True)
class BatchResult:
    """Structured outcome for a directory preprocessing run."""

    input_dir: Path
    output_dir: Path
    config_path: Path
    processed: tuple[FileResult, ...]
    ignored_accel: int
    skipped: tuple[Diagnostic, ...]
    failed: tuple[Diagnostic, ...]
    warnings: tuple[Diagnostic, ...]

    @property
    def warning_count(self) -> int:
        return len(self.warnings) + len(self.skipped)

    @property
    def exit_code(self) -> int:
        return int(bool(self.skipped or self.failed))


@dataclass(frozen=True)
class _ResolvedConfig:
    input_dir: Path
    relative_input_dir: Path
    output_dir: Path
    force: bool


@dataclass(frozen=True)
class _ParsedGyro:
    angular_velocity: FloatArray
    timestamps_ns: npt.NDArray[np.int64]
    physical_row_count: int


@dataclass(frozen=True)
class _IntegratedGyro:
    rows: FloatArray
    sample_interval_seconds: float
    estimated_frequency_hz: float
    fraction_over_ten_percent: float
    max_interval_ratio: float
    unstable: bool


def find_repository_root(start: Path | None = None) -> Path:
    """Find the nearest parent containing this project's package metadata."""
    current = (start or Path.cwd()).resolve()
    candidates = (current, *current.parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "imup"
        ).is_dir():
            return candidate
    raise PreprocessingError(
        "could not find the repository root containing pyproject.toml and src/imup",
        path=current,
    )


def load_preprocess_config(
    config_path: Path,
    *,
    repository_root: Path | None = None,
) -> PreprocessConfig:
    """Load and strictly validate a JSON preprocessing configuration."""
    root = (repository_root or find_repository_root()).resolve()
    resolved_path = config_path if config_path.is_absolute() else root / config_path
    resolved_path = resolved_path.resolve()

    try:
        with resolved_path.open(encoding="utf-8") as config_file:
            payload: object = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise PreprocessingError(
            f"could not read preprocessing JSON: {error}",
            path=resolved_path,
        ) from error

    if not isinstance(payload, dict):
        raise PreprocessingError(
            "preprocessing JSON must contain one object",
            path=resolved_path,
        )
    if not all(isinstance(key, str) for key in payload):
        raise PreprocessingError(
            "preprocessing JSON keys must be strings",
            path=resolved_path,
        )

    values = cast(Mapping[str, object], payload)
    unknown_keys = set(values) - _ALLOWED_CONFIG_KEYS
    if unknown_keys:
        rendered = ", ".join(sorted(unknown_keys))
        raise PreprocessingError(
            f"unknown preprocessing JSON keys: {rendered}",
            path=resolved_path,
        )

    input_value = values.get("input_dir")
    if not isinstance(input_value, str) or not input_value:
        raise PreprocessingError(
            "input_dir must be a non-empty string",
            path=resolved_path,
        )

    force_value = values.get("force", False)
    if not isinstance(force_value, bool):
        raise PreprocessingError(
            "force must be a boolean",
            path=resolved_path,
        )

    return PreprocessConfig(input_dir=Path(input_value), force=force_value)


def preprocess_directory(
    config: PreprocessConfig,
    *,
    repository_root: Path | None = None,
) -> BatchResult:
    """Preprocess direct gyroscope children of one data subdirectory."""
    resolved = _resolve_config(config, repository_root=repository_root)
    gyro_files, ignored_accel, skipped = _discover_inputs(resolved.input_dir)
    failed: list[Diagnostic] = []
    warnings: list[Diagnostic] = []
    processed: list[FileResult] = []

    if not gyro_files:
        failed.append(
            Diagnostic(
                path=resolved.input_dir,
                message="no standard *gyro.csv files were found in this directory",
            )
        )

    resolved.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = resolved.output_dir / _CONFIG_FILENAME
    _write_normalized_config(config_path, resolved)

    for source in gyro_files:
        destination = resolved.output_dir / (
            source.name.removesuffix(_GYRO_SUFFIX) + "integrated_angle.csv"
        )
        if destination.exists() and not resolved.force:
            failed.append(
                Diagnostic(
                    path=destination,
                    message="output already exists; rerun with force=true to replace it",
                )
            )
            continue

        try:
            parsed = _parse_gyro(source)
            integrated = _integrate_gyro(parsed)
            _atomic_write_csv(destination, integrated.rows)
        except PreprocessingError as error:
            failed.append(Diagnostic(path=error.path or source, message=str(error)))
            continue
        except OSError as error:
            failed.append(
                Diagnostic(path=source, message=f"could not write output: {error}")
            )
            continue

        file_result = FileResult(
            source=source,
            destination=destination,
            output_rows=integrated.rows.shape[0],
            sample_interval_seconds=integrated.sample_interval_seconds,
            estimated_frequency_hz=integrated.estimated_frequency_hz,
            fraction_over_ten_percent=integrated.fraction_over_ten_percent,
            max_interval_ratio=integrated.max_interval_ratio,
        )
        processed.append(file_result)
        if integrated.unstable:
            warnings.append(
                Diagnostic(
                    path=source,
                    message=(
                        "sampling intervals are unstable: "
                        f"{integrated.fraction_over_ten_percent:.6%} differ from "
                        "the median by more than 10%, "
                        f"max interval is {integrated.max_interval_ratio:.6f}x "
                        "the median"
                    ),
                )
            )

    return BatchResult(
        input_dir=resolved.input_dir,
        output_dir=resolved.output_dir,
        config_path=config_path,
        processed=tuple(processed),
        ignored_accel=ignored_accel,
        skipped=tuple(skipped),
        failed=tuple(failed),
        warnings=tuple(warnings),
    )


def _resolve_config(
    config: PreprocessConfig,
    *,
    repository_root: Path | None,
) -> _ResolvedConfig:
    if not isinstance(config.force, bool):
        raise PreprocessingError("force must be a boolean")

    root = (repository_root or find_repository_root()).resolve()
    data_root = (root / "data").resolve()
    candidate = config.input_dir
    input_dir = candidate if candidate.is_absolute() else root / candidate
    input_dir = input_dir.resolve()

    if input_dir == data_root or not input_dir.is_relative_to(data_root):
        raise PreprocessingError(
            "input_dir must resolve to a subdirectory below data/",
            path=input_dir,
        )
    if not input_dir.is_dir():
        raise PreprocessingError("input_dir is not a directory", path=input_dir)

    relative_input_dir = input_dir.relative_to(data_root)
    output_dir = root / "outputs" / "preprocessing" / relative_input_dir
    return _ResolvedConfig(
        input_dir=input_dir,
        relative_input_dir=relative_input_dir,
        output_dir=output_dir,
        force=config.force,
    )


def _discover_inputs(
    input_dir: Path,
) -> tuple[list[Path], int, list[Diagnostic]]:
    gyro_files: list[Path] = []
    ignored_accel = 0
    skipped: list[Diagnostic] = []

    for path in sorted(input_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix != ".csv":
            continue
        if path.name.endswith(_GYRO_SUFFIX) and len(path.name) > len(_GYRO_SUFFIX):
            gyro_files.append(path)
        elif path.name.endswith(_ACCEL_SUFFIX) and len(path.name) > len(_ACCEL_SUFFIX):
            ignored_accel += 1
        else:
            skipped.append(
                Diagnostic(
                    path=path,
                    message=(
                        "unsupported CSV filename; expected a non-empty prefix followed "
                        "by gyro.csv or accel.csv; the file was not renamed automatically"
                    ),
                )
            )
    return gyro_files, ignored_accel, skipped


def _parse_gyro(source: Path) -> _ParsedGyro:
    angular_velocity: list[tuple[float, float, float]] = []
    timestamps_ns: list[int] = []
    physical_row_count = 0

    try:
        with source.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            if next(reader, None) is None:
                raise _DataValidationError(
                    "file is empty; the required first physical row is missing",
                    path=source,
                )
            physical_row_count = 1

            for line_number, row in enumerate(reader, start=2):
                physical_row_count += 1
                if len(row) != 4:
                    raise _DataValidationError(
                        f"line {line_number} has {len(row)} columns; expected exactly 4",
                        path=source,
                    )
                try:
                    x, y, z = (float(value) for value in row[:3])
                    timestamp_ns = int(row[3])
                except ValueError as error:
                    raise _DataValidationError(
                        f"line {line_number} contains a non-numeric sensor value",
                        path=source,
                    ) from error
                if not all(math.isfinite(value) for value in (x, y, z)):
                    raise _DataValidationError(
                        f"line {line_number} contains a non-finite angular velocity",
                        path=source,
                    )
                if not 0 <= timestamp_ns < 2**63:
                    raise _DataValidationError(
                        f"line {line_number} timestamp is outside the non-negative "
                        "signed 64-bit range",
                        path=source,
                    )
                if timestamps_ns and timestamp_ns <= timestamps_ns[-1]:
                    raise _DataValidationError(
                        f"line {line_number} timestamp is not strictly increasing",
                        path=source,
                    )
                angular_velocity.append((x, y, z))
                timestamps_ns.append(timestamp_ns)
    except UnicodeDecodeError as error:
        raise _DataValidationError(
            f"file is not valid UTF-8: {error}",
            path=source,
        ) from error
    except csv.Error as error:
        raise _DataValidationError(
            f"could not parse CSV: {error}",
            path=source,
        ) from error
    except OSError as error:
        raise _DataValidationError(
            f"could not read file: {error}",
            path=source,
        ) from error

    if len(angular_velocity) < 2:
        raise _DataValidationError(
            "at least two numeric samples are required after the discarded first row",
            path=source,
        )

    return _ParsedGyro(
        angular_velocity=np.asarray(angular_velocity, dtype=np.float64),
        timestamps_ns=np.asarray(timestamps_ns, dtype=np.int64),
        physical_row_count=physical_row_count,
    )


def _integrate_gyro(parsed: _ParsedGyro) -> _IntegratedGyro:
    deltas_ns = np.diff(parsed.timestamps_ns)
    median_delta_ns = float(np.median(deltas_ns))
    sample_interval_seconds = median_delta_ns / 1_000_000_000
    estimated_frequency_hz = 1.0 / sample_interval_seconds

    relative_deviation = np.abs(deltas_ns - median_delta_ns) / median_delta_ns
    fraction_over_ten_percent = float(np.mean(relative_deviation > 0.10))
    max_interval_ratio = float(np.max(deltas_ns) / median_delta_ns)
    unstable = fraction_over_ten_percent > 0.01 or max_interval_ratio > 2.0

    output_rows = parsed.physical_row_count
    integrated = np.zeros((output_rows, 4), dtype=np.float64)
    increments = parsed.angular_velocity * (180.0 / math.pi) * sample_interval_seconds
    integrated[1:, :3] = np.cumsum(increments, axis=0)
    integrated[:, 3] = np.arange(output_rows, dtype=np.float64) * (
        sample_interval_seconds
    )

    return _IntegratedGyro(
        rows=integrated,
        sample_interval_seconds=sample_interval_seconds,
        estimated_frequency_hz=estimated_frequency_hz,
        fraction_over_ten_percent=fraction_over_ten_percent,
        max_interval_ratio=max_interval_ratio,
        unstable=unstable,
    )


def _write_normalized_config(destination: Path, config: _ResolvedConfig) -> None:
    payload = {
        "input_dir": (Path("data") / config.relative_input_dir).as_posix(),
        "force": config.force,
    }

    if destination.exists() and not config.force:
        try:
            with destination.open(encoding="utf-8") as existing_file:
                existing: object = json.load(existing_file)
        except (OSError, json.JSONDecodeError) as error:
            raise PreprocessingError(
                f"existing normalized config cannot be read: {error}",
                path=destination,
            ) from error
        if existing != payload:
            raise PreprocessingError(
                "normalized config already exists with different parameters; "
                "rerun with force=true to replace it",
                path=destination,
            )
        return

    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(destination, rendered)


def _atomic_write_text(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_csv(destination: Path, rows: FloatArray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.writer(temporary_file)
            writer.writerows(rows.tolist())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
