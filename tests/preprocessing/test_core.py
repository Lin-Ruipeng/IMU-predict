from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from imup.preprocessing import (
    PreprocessConfig,
    PreprocessingError,
    load_preprocess_config,
    preprocess_directory,
)


def _make_repository(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs").mkdir()
    return tmp_path


def _make_input(repository: Path, relative: str = "person/session") -> Path:
    input_dir = repository / "data" / relative
    input_dir.mkdir(parents=True)
    return input_dir


def _write_rows(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows)


def _read_numeric_rows(path: Path) -> list[list[float]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return [[float(value) for value in row] for row in csv.reader(csv_file)]


@pytest.mark.parametrize(
    "first_row",
    [
        [999.0, 999.0, 999.0, 0],
        ["x", "y", "z", "timestamp_ns"],
    ],
)
def test_integrates_after_unconditionally_discarding_first_row(
    tmp_path: Path,
    first_row: list[object],
) -> None:
    repository = _make_repository(tmp_path)
    input_dir = _make_input(repository)
    _write_rows(
        input_dir / "VID_001gyro.csv",
        [
            first_row,
            [1.0, 2.0, 3.0, 10_000_000],
            [2.0, 3.0, 4.0, 20_000_000],
        ],
    )

    result = preprocess_directory(
        PreprocessConfig(input_dir=Path("data/person/session")),
        repository_root=repository,
    )

    assert result.exit_code == 0
    assert len(result.processed) == 1
    file_result = result.processed[0]
    assert file_result.output_rows == 3
    assert file_result.sample_interval_seconds == pytest.approx(0.01)
    assert file_result.estimated_frequency_hz == pytest.approx(100.0)

    rows = _read_numeric_rows(file_result.destination)
    degrees_per_sample = 0.01 * 180.0 / math.pi
    expected = [
        [0.0, 0.0, 0.0, 0.0],
        [degrees_per_sample, 2 * degrees_per_sample, 3 * degrees_per_sample, 0.01],
        [
            3 * degrees_per_sample,
            5 * degrees_per_sample,
            7 * degrees_per_sample,
            0.02,
        ],
    ]
    for actual_row, expected_row in zip(rows, expected, strict=True):
        assert actual_row == pytest.approx(expected_row)
    assert json.loads(result.config_path.read_text(encoding="utf-8")) == {
        "input_dir": "data/person/session",
        "force": False,
    }


def test_scans_only_direct_children_and_preserves_relative_output_path(
    tmp_path: Path,
) -> None:
    repository = _make_repository(tmp_path)
    input_dir = _make_input(repository, "subject/standing")
    nested = input_dir / "nested"
    nested.mkdir()
    rows = [
        ["discarded"],
        [0.0, 0.0, 0.0, 0],
        [0.0, 0.0, 0.0, 1_000_000],
    ]
    _write_rows(input_dir / "VID_directgyro.csv", rows)
    _write_rows(nested / "VID_nestedgyro.csv", rows)

    result = preprocess_directory(
        PreprocessConfig(input_dir=Path("data/subject/standing")),
        repository_root=repository,
    )

    assert [item.source.name for item in result.processed] == ["VID_directgyro.csv"]
    assert result.output_dir == (
        repository / "outputs" / "preprocessing" / "subject" / "standing"
    )
    assert not (result.output_dir / "VID_nestedintegrated_angle.csv").exists()


def test_ignores_accel_and_reports_nonstandard_csv_without_renaming(
    tmp_path: Path,
) -> None:
    repository = _make_repository(tmp_path)
    input_dir = _make_input(repository)
    valid_rows = [
        ["discarded"],
        [0.0, 0.0, 0.0, 0],
        [0.0, 0.0, 0.0, 1_000_000],
    ]
    _write_rows(input_dir / "VID_001gyro.csv", valid_rows)
    _write_rows(input_dir / "VID_001accel.csv", [["not", "read"]])
    _write_rows(input_dir / "VID_002gyro(1).csv", valid_rows)
    _write_rows(input_dir / "notes.csv", [["unsupported"]])

    result = preprocess_directory(
        PreprocessConfig(input_dir=Path("data/person/session")),
        repository_root=repository,
    )

    assert len(result.processed) == 1
    assert result.ignored_accel == 1
    assert {item.path.name for item in result.skipped if item.path is not None} == {
        "VID_002gyro(1).csv",
        "notes.csv",
    }
    assert result.exit_code == 1
    assert not (result.output_dir / "VID_002integrated_angle.csv").exists()


def test_directory_without_standard_gyro_is_an_error(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    input_dir = _make_input(repository)
    _write_rows(input_dir / "VID_001accel.csv", [["ignored"]])

    result = preprocess_directory(
        PreprocessConfig(input_dir=Path("data/person/session")),
        repository_root=repository,
    )

    assert not result.processed
    assert result.ignored_accel == 1
    assert len(result.failed) == 1
    assert "no standard" in result.failed[0].message
    assert result.exit_code == 1


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (["discarded"], "at least two"),
        (["discarded", "1,2,3"], "expected exactly 4"),
        (["discarded", "bad,2,3,1", "1,2,3,2"], "non-numeric"),
        (["discarded", "nan,2,3,1", "1,2,3,2"], "non-finite"),
        (["discarded", "1,2,3,2", "1,2,3,2"], "not strictly increasing"),
        (["discarded", "1,2,3,-1", "1,2,3,2"], "non-negative"),
        (
            ["discarded", f"1,2,3,{2**63}", f"1,2,3,{2**63 + 1}"],
            "signed 64-bit",
        ),
    ],
)
def test_invalid_gyro_fails_atomically(
    tmp_path: Path,
    rows: list[str],
    message: str,
) -> None:
    repository = _make_repository(tmp_path)
    input_dir = _make_input(repository)
    source = input_dir / "VID_invalidgyro.csv"
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = preprocess_directory(
        PreprocessConfig(input_dir=Path("data/person/session")),
        repository_root=repository,
    )

    assert not result.processed
    assert len(result.failed) == 1
    assert message in result.failed[0].message
    assert not (result.output_dir / "VID_invalidintegrated_angle.csv").exists()


def _write_constant_gyro_with_deltas(path: Path, deltas_ns: list[int]) -> None:
    timestamps = [0]
    for delta in deltas_ns:
        timestamps.append(timestamps[-1] + delta)
    rows: list[list[object]] = [["discarded"]]
    rows.extend([[0.0, 0.0, 0.0, timestamp] for timestamp in timestamps])
    _write_rows(path, rows)


@pytest.mark.parametrize(
    "deltas_ns",
    [
        [10_000_000] * 100 + [30_000_000],
        [12_000_000, 12_000_000] + [10_000_000] * 98,
    ],
)
def test_unstable_sampling_warns_but_succeeds(
    tmp_path: Path,
    deltas_ns: list[int],
) -> None:
    repository = _make_repository(tmp_path)
    input_dir = _make_input(repository)
    _write_constant_gyro_with_deltas(input_dir / "VID_unstablegyro.csv", deltas_ns)

    result = preprocess_directory(
        PreprocessConfig(input_dir=Path("data/person/session")),
        repository_root=repository,
    )

    assert len(result.processed) == 1
    assert len(result.warnings) == 1
    assert "unstable" in result.warnings[0].message
    assert result.exit_code == 0


def test_constant_unusual_absolute_frequency_is_accepted(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    input_dir = _make_input(repository)
    _write_constant_gyro_with_deltas(
        input_dir / "VID_slowgyro.csv",
        [1_000_000_000] * 4,
    )

    result = preprocess_directory(
        PreprocessConfig(input_dir=Path("data/person/session")),
        repository_root=repository,
    )

    assert result.processed[0].estimated_frequency_hz == pytest.approx(1.0)
    assert not result.warnings
    assert result.exit_code == 0


def test_partial_failure_keeps_valid_output_and_returns_nonzero(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    input_dir = _make_input(repository)
    _write_constant_gyro_with_deltas(
        input_dir / "VID_goodgyro.csv",
        [10_000_000, 10_000_000],
    )
    _write_rows(
        input_dir / "VID_badgyro.csv",
        [["discarded"], [0.0, 0.0, 0.0, 1]],
    )

    result = preprocess_directory(
        PreprocessConfig(input_dir=Path("data/person/session")),
        repository_root=repository,
    )

    assert [item.source.name for item in result.processed] == ["VID_goodgyro.csv"]
    assert len(result.failed) == 1
    assert result.exit_code == 1


def test_existing_output_requires_force_and_force_rewrites_config(
    tmp_path: Path,
) -> None:
    repository = _make_repository(tmp_path)
    input_dir = _make_input(repository)
    source = input_dir / "VID_001gyro.csv"
    _write_constant_gyro_with_deltas(source, [10_000_000, 10_000_000])
    config = PreprocessConfig(input_dir=Path("data/person/session"))

    first = preprocess_directory(config, repository_root=repository)
    second = preprocess_directory(config, repository_root=repository)
    forced = preprocess_directory(
        PreprocessConfig(input_dir=config.input_dir, force=True),
        repository_root=repository,
    )

    assert len(first.processed) == 1
    assert not second.processed
    assert len(second.failed) == 1
    assert second.exit_code == 1
    assert len(forced.processed) == 1
    assert json.loads(forced.config_path.read_text(encoding="utf-8"))["force"] is True


def test_same_filename_in_different_inputs_has_isolated_outputs(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    first_input = _make_input(repository, "first/session")
    second_input = _make_input(repository, "second/session")
    _write_constant_gyro_with_deltas(
        first_input / "VID_samegyro.csv",
        [10_000_000, 10_000_000],
    )
    _write_constant_gyro_with_deltas(
        second_input / "VID_samegyro.csv",
        [20_000_000, 20_000_000],
    )

    first = preprocess_directory(
        PreprocessConfig(input_dir=Path("data/first/session")),
        repository_root=repository,
    )
    second = preprocess_directory(
        PreprocessConfig(input_dir=Path("data/second/session")),
        repository_root=repository,
    )

    assert first.processed[0].destination != second.processed[0].destination
    assert first.processed[0].destination.exists()
    assert second.processed[0].destination.exists()


def test_loads_json_config_relative_to_repository_root(tmp_path: Path) -> None:
    repository = _make_repository(tmp_path)
    config_path = repository / "outputs" / "request.json"
    config_path.write_text(
        json.dumps({"input_dir": "data/person/session", "force": True}),
        encoding="utf-8",
    )

    config = load_preprocess_config(
        Path("outputs/request.json"),
        repository_root=repository,
    )

    assert config == PreprocessConfig(
        input_dir=Path("data/person/session"),
        force=True,
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (["not", "an", "object"], "one object"),
        ({"force": False}, "input_dir"),
        ({"input_dir": "data/x", "force": "yes"}, "boolean"),
        ({"input_dir": "data/x", "extra": 1}, "unknown"),
    ],
)
def test_rejects_invalid_json_config(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    repository = _make_repository(tmp_path)
    config_path = repository / "outputs" / "request.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreprocessingError, match=message):
        load_preprocess_config(config_path, repository_root=repository)


@pytest.mark.parametrize("input_dir", [Path("data"), Path("outside")])
def test_rejects_input_outside_a_data_subdirectory(
    tmp_path: Path,
    input_dir: Path,
) -> None:
    repository = _make_repository(tmp_path)
    (repository / "outside").mkdir()

    with pytest.raises(PreprocessingError, match="subdirectory below data"):
        preprocess_directory(
            PreprocessConfig(input_dir=input_dir),
            repository_root=repository,
        )
