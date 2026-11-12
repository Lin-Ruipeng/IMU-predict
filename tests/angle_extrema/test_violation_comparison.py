from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from imup import main
from imup.angle_extrema.violation_comparison import _prepare_window


def _repository(tmp_path: Path) -> Path:
    (tmp_path / "src" / "imup").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "outputs" / "preprocessing").mkdir(parents=True)
    return tmp_path


def _write_rows(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows.tolist())


def _write_fixture(repository: Path) -> None:
    elapsed = np.arange(0.0, 2.1, 0.1)
    raw_angles = np.column_stack(
        [2.0 * elapsed, np.sin(elapsed), np.zeros(elapsed.shape)]
    )
    target_angles = np.column_stack(
        [2.0 * elapsed, np.sin(elapsed - 0.1), np.zeros(elapsed.shape)]
    )
    source_relative = Path(
        "outputs/preprocessing/person/session/sampleintegrated_angle.csv"
    )
    source = repository / source_relative
    _write_rows(source, np.column_stack([raw_angles, elapsed]))

    analysis_run = repository / "outputs/angle_extrema/runs/source-run"
    target_path = (
        analysis_run
        / "sources/person/session/sampleintegrated_angle/filtered/lowpass_11hz.csv"
    )
    _write_rows(target_path, np.column_stack([target_angles, elapsed]))
    (analysis_run / "analysis_config.json").write_text(
        json.dumps(
            {
                "run_name": "source-run",
                "input_paths": [source_relative.as_posix()],
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
    target_violation = {
        "source_file": source_relative.as_posix(),
        "signal_variant": "lowpass_11hz",
        "window_index": 0,
        "start_row": 0,
        "start_elapsed_s": 0.0,
        "end_elapsed_s_exclusive": 1.0,
        "axis_extrema": {
            "x": {"min_deg": 0.0, "max_deg": 1.8},
            "y": {"min_deg": 0.0, "max_deg": 0.8},
            "z": {"min_deg": 0.0, "max_deg": 0.0},
        },
        "violations": {"x": ["above_upper_bound"]},
    }
    ignored_raw_violation = dict(target_violation)
    ignored_raw_violation["signal_variant"] = "raw"
    ignored_raw_violation["window_index"] = 99
    (analysis_run / "extrema_results.json").write_text(
        json.dumps(
            {
                "run_status": "succeeded",
                "summaries": [],
                "violations": [ignored_raw_violation, target_violation],
                "failures": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    comparison = repository / "outputs/angle_extrema/comparisons/known-lag"
    comparison.mkdir(parents=True)
    (comparison / "comparison_config.json").write_text(
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
    with (comparison / "delay_compensation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "source_file",
                "comparator",
                "target",
                "estimated_shift_s",
                "estimated_shift_ms",
                "shift_action",
                "correlation",
                "overlap_start_s",
                "overlap_end_s",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_file": source_relative.as_posix(),
                "comparator": "raw",
                "target": "lowpass_11hz",
                "estimated_shift_s": 0.1,
                "estimated_shift_ms": 100.0,
                "shift_action": "delay",
                "correlation": 0.99,
                "overlap_start_s": 0.1,
                "overlap_end_s": 2.0,
            }
        )


def test_compare_violations_uses_only_target_windows_and_reports_partial_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _repository(tmp_path)
    _write_fixture(repository)
    monkeypatch.chdir(repository)

    exit_code = main(
        [
            "extrema",
            "compare-violations",
            "--analysis-run-name",
            "source-run",
            "--comparison-name",
            "known-lag",
        ]
    )

    output = (
        repository
        / "outputs/angle_extrema/comparisons/known-lag/target_violation_windows"
    )
    plot = (
        output
        / "sources/person/session/sampleintegrated_angle/raw"
        / "window_000000_start_0s.png"
    )
    with (output / "window_comparisons.csv").open(
        encoding="utf-8", newline=""
    ) as csv_file:
        records = list(csv.DictReader(csv_file))
    results = json.loads(
        (output / "violation_comparison_results.json").read_text(encoding="utf-8")
    )
    stderr = capsys.readouterr().err

    assert exit_code == 0
    assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(records) == 1
    assert records[0]["target_window_index"] == "0"
    assert records[0]["coverage_status"] == "partial"
    assert records[0]["violated_axes"] == "x"
    assert results["target_violation_windows"] == 1
    assert results["plots"] == 1
    assert results["partial_coverages"] == 1
    assert "PARTIAL_COVERAGE" in stderr
    assert "SUMMARY: processed=1 target_windows=1 plots=1 failed=0 warnings=1" in stderr


def test_prepare_window_rebases_full_and_partial_curves_to_their_first_point() -> None:
    elapsed = np.arange(0.0, 1.1, 0.1)
    target = np.column_stack([elapsed, 2.0 * elapsed, -elapsed, elapsed])
    comparator = np.column_stack(
        [10.0 + elapsed, 20.0 + 2.0 * elapsed, 30.0 - elapsed, elapsed]
    )

    prepared = _prepare_window(
        target,
        comparator,
        start_row=0,
        end_elapsed_s_exclusive=1.0,
        shift_s=0.1,
    )

    assert prepared.target_time_s[0] == 0.0
    assert prepared.target_angles_deg[0] == pytest.approx([0.0, 0.0, 0.0])
    assert prepared.comparator_time_s[0] == pytest.approx(0.1)
    assert prepared.comparator_angles_deg[0] == pytest.approx([0.0, 0.0, 0.0])
    assert prepared.coverage_status == "partial"


def test_compare_violations_accepts_a_complete_json_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    _write_fixture(repository)
    request = repository / "outputs/violation-comparison-request.json"
    request.write_text(
        json.dumps(
            {
                "comparison_name": "known-lag",
                "analysis_run_name": "source-run",
                "force": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(repository)

    exit_code = main(["extrema", "compare-violations", "--config", str(request)])

    saved_config = json.loads(
        (
            repository
            / "outputs/angle_extrema/comparisons/known-lag"
            / "target_violation_windows/violation_comparison_config.json"
        ).read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert saved_config == {
        "comparison_name": "known-lag",
        "analysis_run_name": "source-run",
        "force": False,
    }
