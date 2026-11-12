from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from imup import main


def _repository(tmp_path: Path) -> Path:
    (tmp_path / "src" / "imup").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return tmp_path


def _write_rows(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows.tolist())


def test_compare_aligns_one_source_and_records_the_shared_tri_axis_shift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _repository(tmp_path)
    elapsed = np.arange(0.0, 6.0, 0.01)
    base = np.column_stack(
        [
            np.sin(2 * np.pi * 1.1 * elapsed) + 0.3 * np.sin(2 * np.pi * 2.7 * elapsed),
            0.7 * np.sin(2 * np.pi * 0.8 * elapsed + 0.4),
            0.5 * np.cos(2 * np.pi * 1.7 * elapsed - 0.2),
        ]
    )
    shift_samples = 4
    target = np.vstack(
        [np.repeat(base[[0]], shift_samples, axis=0), base[:-shift_samples]]
    )
    source = repository / "outputs/preprocessing/session/sampleintegrated_angle.csv"
    _write_rows(source, np.column_stack([base, elapsed]))

    run_dir = repository / "outputs/angle_extrema/runs/source-run"
    filtered = run_dir / "sources/session/sampleintegrated_angle/filtered"
    _write_rows(filtered / "lowpass_11hz.csv", np.column_stack([target, elapsed]))
    (run_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "run_name": "source-run",
                "input_paths": [
                    "outputs/preprocessing/session/sampleintegrated_angle.csv"
                ],
                "window_duration_s": 1.0,
                "stride_duration_s": 0.5,
                "lower_bound_deg": -1.0,
                "upper_bound_deg": 1.0,
                "cutoff_frequencies_hz": [11.0],
                "violation_plot_input_paths": [],
                "save_filtered_csv": True,
                "allow_external_input": False,
                "force": False,
            }
        ),
        encoding="utf-8",
    )
    request = repository / "outputs/comparison-request.json"
    request.write_text(
        json.dumps(
            {
                "comparison_name": "known-lag",
                "analysis_run_name": "source-run",
                "target_cutoff_hz": 11.0,
                "comparison_cutoff_frequencies_hz": [],
                "include_raw": True,
                "force": False,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(repository)
    exit_code = main(["extrema", "compare", "--config", str(request)])

    output = repository / "outputs/angle_extrema/comparisons/known-lag"
    source_output = output / "sources/session/sampleintegrated_angle"
    with (output / "delay_compensation.csv").open(
        encoding="utf-8", newline=""
    ) as csv_file:
        records = list(csv.DictReader(csv_file))
    stderr = capsys.readouterr().err

    assert exit_code == 0
    assert len(records) == 1
    assert records[0]["comparator"] == "raw"
    assert float(records[0]["estimated_shift_ms"]) == pytest.approx(40.0, abs=5.1)
    assert records[0]["shift_action"] == "delay"
    assert float(records[0]["overlap_start_s"]) == pytest.approx(0.04, abs=0.011)
    assert float(records[0]["overlap_end_s"]) == pytest.approx(5.99)
    assert (
        source_output / "raw_vs_target_11hz_without_delay_compensation.png"
    ).is_file()
    assert (source_output / "raw_vs_target_11hz_with_delay_compensation.png").is_file()
    assert (output / "comparison_config.json").is_file()
    assert (output / "README.md").is_file()
    assert "SUMMARY: processed=1 comparisons=1 failed=0" in stderr


def test_compare_uses_informative_axes_when_one_axis_is_constant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    elapsed = np.arange(0.0, 4.0, 0.01)
    comparator = np.column_stack(
        [
            np.sin(2 * np.pi * 1.3 * elapsed),
            np.cos(2 * np.pi * 0.7 * elapsed),
            np.full(elapsed.shape, 2.0),
        ]
    )
    shift_samples = 3
    target = np.vstack(
        [
            np.repeat(comparator[[0]], shift_samples, axis=0),
            comparator[:-shift_samples],
        ]
    )
    source = repository / "outputs/preprocessing/session/sampleintegrated_angle.csv"
    _write_rows(source, np.column_stack([comparator, elapsed]))
    run_dir = repository / "outputs/angle_extrema/runs/source-run"
    _write_rows(
        run_dir / "sources/session/sampleintegrated_angle/filtered/lowpass_11hz.csv",
        np.column_stack([target, elapsed]),
    )
    (run_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "run_name": "source-run",
                "input_paths": [str(source.relative_to(repository))],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(repository)
    exit_code = main(
        [
            "extrema",
            "compare",
            "--analysis-run-name",
            "source-run",
            "--comparison-name",
            "constant-axis",
            "--comparison-cutoff-frequencies-hz",
        ]
    )

    output = repository / "outputs/angle_extrema/comparisons/constant-axis"
    with (output / "delay_compensation.csv").open(
        encoding="utf-8", newline=""
    ) as csv_file:
        record = next(csv.DictReader(csv_file))

    assert exit_code == 0
    assert float(record["estimated_shift_ms"]) == pytest.approx(30.0, abs=5.1)


def test_compare_reports_every_missing_comparator_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _repository(tmp_path)
    elapsed = np.arange(0.0, 1.0, 0.01)
    angles = np.column_stack(
        [
            np.sin(2 * np.pi * elapsed),
            np.cos(2 * np.pi * elapsed),
            np.sin(4 * np.pi * elapsed),
        ]
    )
    source = repository / "outputs/preprocessing/session/sampleintegrated_angle.csv"
    _write_rows(source, np.column_stack([angles, elapsed]))
    run_dir = repository / "outputs/angle_extrema/runs/source-run"
    _write_rows(
        run_dir / "sources/session/sampleintegrated_angle/filtered/lowpass_11hz.csv",
        np.column_stack([angles, elapsed]),
    )
    (run_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "run_name": "source-run",
                "input_paths": [str(source.relative_to(repository))],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(repository)
    exit_code = main(
        [
            "extrema",
            "compare",
            "--analysis-run-name",
            "source-run",
            "--comparison-name",
            "missing-variants",
            "--no-include-raw",
            "--comparison-cutoff-frequencies-hz",
            "3",
            "5",
        ]
    )

    stderr = capsys.readouterr().err

    assert exit_code == 1
    assert "lowpass_3hz.csv" in stderr
    assert "lowpass_5hz.csv" in stderr
    assert "SUMMARY: processed=0 comparisons=0 failed=2" in stderr
