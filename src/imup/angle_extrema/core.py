from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from imup.preprocessing import find_repository_root

_ANGLE_SUFFIX = "integrated_angle.csv"
_ALLOWED_SELECT_KEYS = frozenset({"input_root", "output", "quantile", "force"})


class AngleExtremaError(Exception):
    """Base error for invalid angle-extrema requests."""

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        super().__init__(message)
        self.path = path


@dataclass(frozen=True)
class SelectConfig:
    """Configuration for representative-file selection."""

    input_root: Path
    output: Path
    quantile: float = 0.5
    force: bool = False


@dataclass(frozen=True)
class SelectionDetail:
    """One directory's representative integrated-angle file."""

    directory: Path
    directory_file_count: int
    quantile: float
    row_count: int
    selected_path: Path


@dataclass(frozen=True)
class SelectionWarning:
    """One path-scoped representative-selection warning."""

    path: Path
    message: str


@dataclass(frozen=True)
class SelectionResult:
    """Observable result of one selection run."""

    output_path: Path
    details: tuple[SelectionDetail, ...]
    warnings: tuple[SelectionWarning, ...]


def load_select_config(
    config_path: Path,
    *,
    repository_root: Path | None = None,
) -> SelectConfig:
    """Load and strictly validate a representative-selection JSON request."""
    root = (repository_root or find_repository_root()).resolve()
    resolved_path = _resolve_path(config_path, root)
    try:
        with resolved_path.open(encoding="utf-8") as config_file:
            payload: object = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise AngleExtremaError(
            f"could not read selection JSON: {error}",
            path=resolved_path,
        ) from error
    if not isinstance(payload, dict):
        raise AngleExtremaError(
            "selection JSON must contain one object", path=resolved_path
        )
    if not all(isinstance(key, str) for key in payload):
        raise AngleExtremaError(
            "selection JSON keys must be strings", path=resolved_path
        )
    values = cast(Mapping[str, object], payload)
    unknown = set(values) - _ALLOWED_SELECT_KEYS
    if unknown:
        raise AngleExtremaError(
            f"unknown selection JSON keys: {', '.join(sorted(unknown))}",
            path=resolved_path,
        )
    input_root = values.get("input_root")
    output = values.get("output")
    if not isinstance(input_root, str) or not input_root:
        raise AngleExtremaError(
            "input_root must be a non-empty string", path=resolved_path
        )
    if not isinstance(output, str) or not output:
        raise AngleExtremaError("output must be a non-empty string", path=resolved_path)
    quantile = values.get("quantile", 0.5)
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
        raise AngleExtremaError("quantile must be a number", path=resolved_path)
    force = values.get("force", False)
    if not isinstance(force, bool):
        raise AngleExtremaError("force must be a boolean", path=resolved_path)
    return SelectConfig(
        input_root=Path(input_root),
        output=Path(output),
        quantile=float(quantile),
        force=force,
    )


def select_representative_files(
    config: SelectConfig,
    *,
    repository_root: Path | None = None,
) -> SelectionResult:
    """Select one nearest-rank file by line-count quantile per directory."""
    root = (repository_root or find_repository_root()).resolve()
    input_root = _resolve_path(config.input_root, root)
    output_path = _resolve_path(config.output, root)

    if not isinstance(config.force, bool):
        raise AngleExtremaError("force must be a boolean")
    if (
        isinstance(config.quantile, bool)
        or not isinstance(config.quantile, (int, float))
        or not math.isfinite(config.quantile)
        or not 0.0 < config.quantile <= 1.0
    ):
        raise AngleExtremaError("quantile must be finite and in the interval (0, 1]")
    if not input_root.is_dir():
        raise AngleExtremaError("input_root is not a directory", path=input_root)
    preprocessing_root = (root / "outputs" / "preprocessing").resolve()
    if not input_root.is_relative_to(preprocessing_root):
        raise AngleExtremaError(
            "input_root must resolve inside outputs/preprocessing",
            path=input_root,
        )
    selection_root = (root / "outputs" / "angle_extrema" / "selections").resolve()
    if not output_path.is_relative_to(selection_root):
        raise AngleExtremaError(
            "output must resolve inside outputs/angle_extrema/selections",
            path=output_path,
        )
    if output_path.exists() and not config.force:
        raise AngleExtremaError(
            "selection output already exists; rerun with force=true to replace it",
            path=output_path,
        )

    grouped: dict[Path, list[Path]] = defaultdict(list)
    for candidate in input_root.rglob(f"*{_ANGLE_SUFFIX}"):
        if candidate.is_file() and len(candidate.name) > len(_ANGLE_SUFFIX):
            grouped[candidate.parent].append(candidate)

    if not grouped:
        raise AngleExtremaError(
            "no *integrated_angle.csv files were found below input_root",
            path=input_root,
        )

    details: list[SelectionDetail] = []
    warnings: list[SelectionWarning] = []
    for directory in sorted(grouped, key=lambda item: item.relative_to(input_root)):
        ranked = sorted(
            ((_count_rows(path), path) for path in grouped[directory]),
            key=lambda item: (item[0], item[1].relative_to(root).as_posix()),
        )
        index = math.ceil(float(config.quantile) * len(ranked)) - 1
        row_count, selected = ranked[index]
        if len(ranked) < 3:
            warnings.append(
                SelectionWarning(
                    path=directory,
                    message="directory contains fewer than three angle files",
                )
            )
        details.append(
            SelectionDetail(
                directory=directory,
                directory_file_count=len(ranked),
                quantile=float(config.quantile),
                row_count=row_count,
                selected_path=selected,
            )
        )

    payload = {
        "selection_config": {
            "input_root": input_root.relative_to(root).as_posix(),
            "quantile": float(config.quantile),
            "output": output_path.relative_to(root).as_posix(),
            "force": config.force,
        },
        "selected_input_paths": [
            item.selected_path.relative_to(root).as_posix() for item in details
        ],
        "selection_details": [
            {
                "directory": item.directory.relative_to(root).as_posix(),
                "directory_file_count": item.directory_file_count,
                "quantile": item.quantile,
                "row_count": item.row_count,
                "selected_path": item.selected_path.relative_to(root).as_posix(),
            }
            for item in details
        ],
        "warnings": [
            {
                "path": item.path.relative_to(root).as_posix(),
                "message": item.message,
            }
            for item in warnings
        ],
    }
    _atomic_write_text(
        output_path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    return SelectionResult(
        output_path=output_path,
        details=tuple(details),
        warnings=tuple(warnings),
    )


def _resolve_path(path: Path, root: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def _count_rows(path: Path) -> int:
    try:
        with path.open(encoding="utf-8", newline="") as source:
            return sum(1 for _ in source)
    except (OSError, UnicodeDecodeError) as error:
        raise AngleExtremaError(f"could not count rows: {error}", path=path) from error


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
