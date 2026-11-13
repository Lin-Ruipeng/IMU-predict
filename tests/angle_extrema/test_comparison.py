from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from imup import main
from imup.angle_extrema.comparison import (
    ComparisonConfig,
    _align_comparator,
    _anchor_relative_absolute_errors,
    _empirical_quantiles,
    load_comparison_config,
    run_angle_comparison,
)
from imup.angle_extrema.core import AngleExtremaError


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
    for cutoff_hz in (3, 5, 7, 9):
        _write_rows(
            filtered / f"lowpass_{cutoff_hz}hz.csv",
            np.column_stack([base, elapsed]),
        )
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
                "comparison_cutoff_frequencies_hz": [3.0, 5.0, 7.0, 9.0],
                "include_raw": True,
                "error_ecdf": True,
                "error_ecdf_percentile_guides": True,
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
    assert len(records) == 5
    assert {record["comparator"] for record in records} == {
        "raw",
        "lowpass_3hz",
        "lowpass_5hz",
        "lowpass_7hz",
        "lowpass_9hz",
    }
    assert records[0]["comparator"] == "raw"
    assert float(records[0]["estimated_shift_ms"]) == pytest.approx(40.0, abs=5.1)
    assert records[0]["shift_action"] == "delay"
    assert float(records[0]["overlap_start_s"]) == pytest.approx(0.04, abs=0.011)
    assert float(records[0]["overlap_end_s"]) == pytest.approx(5.99)
    assert (
        source_output / "raw_vs_target_11hz_without_delay_compensation.png"
    ).is_file()
    assert (source_output / "raw_vs_target_11hz_with_delay_compensation.png").is_file()
    assert (source_output / "absolute_error_ecdf.png").is_file()
    with (source_output / "absolute_errors.csv").open(
        encoding="utf-8", newline=""
    ) as csv_file:
        error_records = list(csv.DictReader(csv_file))
    assert error_records[0].keys() == {
        "target_elapsed_s",
        "comparator",
        "x_error_deg",
        "y_error_deg",
        "z_error_deg",
    }
    assert {record["comparator"] for record in error_records} == {
        "raw",
        "lowpass_3hz",
        "lowpass_5hz",
        "lowpass_7hz",
        "lowpass_9hz",
    }
    with (output / "absolute_error_summary.csv").open(
        encoding="utf-8", newline=""
    ) as csv_file:
        summary_records = list(csv.DictReader(csv_file))
    assert len(summary_records) == 15
    assert {record["axis"] for record in summary_records} == {"x", "y", "z"}
    normalized_config = json.loads(
        (output / "comparison_config.json").read_text(encoding="utf-8")
    )
    assert normalized_config["error_ecdf"] is True
    assert normalized_config["error_ecdf_percentile_guides"] is True
    assert (output / "comparison_config.json").is_file()
    assert (output / "README.md").is_file()
    assert "SUMMARY: processed=1 comparisons=5 failed=0" in stderr


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
            "--error-ecdf",
            "--error-ecdf-percentile-guides",
        ]
    )

    output = repository / "outputs/angle_extrema/comparisons/constant-axis"
    with (output / "delay_compensation.csv").open(
        encoding="utf-8", newline=""
    ) as csv_file:
        record = next(csv.DictReader(csv_file))

    assert exit_code == 0
    assert float(record["estimated_shift_ms"]) == pytest.approx(30.0, abs=5.1)
    assert (
        output / "sources/session/sampleintegrated_angle/absolute_error_ecdf.png"
    ).is_file()


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
    output = repository / "outputs/angle_extrema/comparisons/missing-variants"
    assert not (output / "absolute_error_summary.csv").exists()


def test_fractional_alignment_uses_linear_interpolation_before_anchor_zeroing() -> None:
    elapsed = np.arange(4.0)
    target_angles = np.column_stack(
        [
            [0.0, 2.0, 6.0, 10.0],
            [5.0, 8.0, 13.0, 20.0],
            [3.0, 2.0, 0.0, -3.0],
        ]
    )
    comparator_angles = np.column_stack(
        [
            10.0 + 2.0 * elapsed,
            -4.0 + 3.0 * elapsed,
            8.0 - elapsed,
        ]
    )
    target_rows = np.column_stack([target_angles, elapsed])
    comparator_rows = np.column_stack([comparator_angles, elapsed])

    aligned_time, aligned_target, aligned_comparator = _align_comparator(
        target_rows,
        comparator_rows,
        shift_s=0.5,
    )
    errors = _anchor_relative_absolute_errors(aligned_target, aligned_comparator)

    np.testing.assert_allclose(aligned_time, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        aligned_comparator,
        [
            [11.0, -2.5, 7.5],
            [13.0, 0.5, 6.5],
            [15.0, 3.5, 5.5],
        ],
    )
    np.testing.assert_allclose(
        errors,
        [
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 1.0],
            [4.0, 6.0, 3.0],
        ],
    )


def test_empirical_quantiles_are_observed_inverse_ecdf_values() -> None:
    values = np.asarray([0.0, 1.0, 10.0, 20.0])

    quantiles = _empirical_quantiles(values)

    np.testing.assert_allclose(quantiles, [1.0, 20.0, 20.0, 20.0])


def test_comparison_json_rejects_non_boolean_error_ecdf_options(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    request = repository / "outputs/comparison-request.json"
    request.parent.mkdir(parents=True)
    request.write_text(
        json.dumps(
            {
                "comparison_name": "invalid-ecdf",
                "analysis_run_name": "source-run",
                "error_ecdf": "yes",
                "error_ecdf_percentile_guides": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AngleExtremaError, match="error_ecdf must be a boolean"):
        load_comparison_config(request, repository_root=repository)


def test_percentile_guides_require_error_ecdf(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    config = ComparisonConfig(
        comparison_name="invalid-guides",
        analysis_run_name="source-run",
        error_ecdf=False,
        error_ecdf_percentile_guides=True,
    )

    with pytest.raises(
        AngleExtremaError,
        match="error_ecdf_percentile_guides requires error_ecdf=true",
    ):
        run_angle_comparison(config, repository_root=repository)
