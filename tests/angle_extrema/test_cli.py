from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from imup import main


def _make_repository(tmp_path: Path) -> Path:
    (tmp_path / "src" / "imup").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "outputs" / "preprocessing").mkdir(parents=True)
    return tmp_path


def _write_angle_file(path: Path, row_count: int, *, dt: float = 0.01) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for index in range(row_count):
            writer.writerow([float(index), 0.0, 0.0, index * dt])


def test_select_reports_one_nearest_rank_median_file_per_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _make_repository(tmp_path)
    preprocessing = repository / "outputs" / "preprocessing"

    first = preprocessing / "first"
    _write_angle_file(first / "shortintegrated_angle.csv", 2)
    _write_angle_file(first / "medianintegrated_angle.csv", 4)
    _write_angle_file(first / "longintegrated_angle.csv", 6)

    second = preprocessing / "nested" / "second"
    _write_angle_file(second / "a_integrated_angle.csv", 2)
    _write_angle_file(second / "b_integrated_angle.csv", 4)
    _write_angle_file(second / "c_integrated_angle.csv", 6)
    _write_angle_file(second / "d_integrated_angle.csv", 8)

    monkeypatch.chdir(repository)
    exit_code = main(
        [
            "extrema",
            "select",
            "--input-root",
            "outputs/preprocessing",
            "--quantile",
            "0.5",
            "--output",
            "outputs/angle_extrema/selections/q50.json",
        ]
    )

    report_path = repository / "outputs" / "angle_extrema" / "selections/q50.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["selected_input_paths"] == [
        "outputs/preprocessing/first/medianintegrated_angle.csv",
        "outputs/preprocessing/nested/second/b_integrated_angle.csv",
    ]
    assert [item["row_count"] for item in report["selection_details"]] == [4, 4]
    assert [item["directory_file_count"] for item in report["selection_details"]] == [
        3,
        4,
    ]
    assert all(item["quantile"] == 0.5 for item in report["selection_details"])
    assert report["selection_config"] == {
        "input_root": "outputs/preprocessing",
        "quantile": 0.5,
        "output": "outputs/angle_extrema/selections/q50.json",
        "force": False,
    }
    assert "SUMMARY: directories=2 selected=2 warnings=0" in capsys.readouterr().err


def test_select_accepts_a_strict_json_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _make_repository(tmp_path)
    input_dir = repository / "outputs" / "preprocessing" / "session"
    _write_angle_file(input_dir / "shortintegrated_angle.csv", 2)
    _write_angle_file(input_dir / "longintegrated_angle.csv", 4)
    request = repository / "outputs" / "select-request.json"
    request.write_text(
        json.dumps(
            {
                "input_root": "outputs/preprocessing",
                "quantile": 0.5,
                "output": "outputs/angle_extrema/selections/from-config.json",
                "force": False,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(repository)
    exit_code = main(["extrema", "select", "--config", "outputs/select-request.json"])

    report = json.loads(
        (
            repository / "outputs" / "angle_extrema" / "selections" / "from-config.json"
        ).read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert report["selected_input_paths"] == [
        "outputs/preprocessing/session/shortintegrated_angle.csv"
    ]


def test_run_uses_fixed_physical_windows_and_records_extrema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _make_repository(tmp_path)
    source = (
        repository
        / "outputs"
        / "preprocessing"
        / "person"
        / "session"
        / "sampleintegrated_angle.csv"
    )
    source.parent.mkdir(parents=True)
    with source.open("w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file).writerows(
            [
                [10.0, 0.0, 0.0, 0.00],
                [10.5, 0.0, 0.0, 0.25],
                [11.2, 0.0, 0.0, 0.50],
                [9.5, 0.0, 0.0, 0.75],
                [9.4, 0.0, 0.0, 1.00],
                [10.6, 0.0, 0.0, 1.25],
                [10.5, 0.0, 0.0, 1.50],
            ]
        )
    config_path = repository / "outputs" / "run-request.json"
    config_path.write_text(
        json.dumps(
            {
                "run_name": "raw-only",
                "input_paths": [
                    "outputs/preprocessing/person/session/sampleintegrated_angle.csv"
                ],
                "window_duration_s": 1.0,
                "stride_duration_s": 0.5,
                "lower_bound_deg": -1.0,
                "upper_bound_deg": 1.0,
                "cutoff_frequencies_hz": [],
                "violation_plot_input_paths": [],
                "save_filtered_csv": True,
                "allow_external_input": False,
                "force": False,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(repository)
    exit_code = main(["extrema", "run", "--config", "outputs/run-request.json"])

    run_dir = repository / "outputs" / "angle_extrema" / "runs" / "raw-only"
    results = json.loads((run_dir / "extrema_results.json").read_text(encoding="utf-8"))
    saved_config = json.loads(
        (run_dir / "analysis_config.json").read_text(encoding="utf-8")
    )
    source_dir = run_dir / "sources" / "person" / "session" / "sampleintegrated_angle"
    raw_plot = source_dir / "raw_series.png"
    violation_timeline = source_dir / "raw_violation_timeline.png"

    assert exit_code == 0
    assert results["run_status"] == "succeeded"
    assert results["failures"] == []
    assert results["summaries"] == [
        {
            "source_file": (
                "outputs/preprocessing/person/session/sampleintegrated_angle.csv"
            ),
            "signal_variant": "raw",
            "sample_interval_s": 0.25,
            "estimated_frequency_hz": 4.0,
            "total_windows": 2,
            "violating_windows": 2,
        }
    ]
    assert [item["start_row"] for item in results["violations"]] == [0, 2]
    assert [item["start_elapsed_s"] for item in results["violations"]] == [
        0.0,
        0.5,
    ]
    assert [item["end_elapsed_s_exclusive"] for item in results["violations"]] == [
        1.0,
        1.5,
    ]
    assert results["violations"][0]["axis_extrema"]["x"] == {
        "min_deg": -0.5,
        "max_deg": 1.2,
    }
    assert results["violations"][0]["violations"] == {"x": ["above_upper_bound"]}
    assert results["violations"][1]["violations"] == {"x": ["below_lower_bound"]}
    assert saved_config["input_paths"] == [
        "outputs/preprocessing/person/session/sampleintegrated_angle.csv"
    ]
    assert saved_config["cutoff_frequencies_hz"] == []
    assert raw_plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert violation_timeline.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "SUMMARY: processed=1 failed=0" in capsys.readouterr().err


def test_run_saves_a_causal_lowpass_variant_with_steady_initial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _make_repository(tmp_path)
    source = (
        repository
        / "outputs"
        / "preprocessing"
        / "person"
        / "session"
        / "stepintegrated_angle.csv"
    )
    source.parent.mkdir(parents=True)
    with source.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for index in range(201):
            writer.writerow([0.0 if index < 100 else 1.0, 2.0, 0.0, index * 0.01])
    config_path = repository / "outputs" / "filter-request.json"
    config_path.write_text(
        json.dumps(
            {
                "run_name": "causal-filter",
                "input_paths": [
                    "outputs/preprocessing/person/session/stepintegrated_angle.csv"
                ],
                "window_duration_s": 1.0,
                "stride_duration_s": 0.5,
                "lower_bound_deg": -0.5,
                "upper_bound_deg": 0.5,
                "cutoff_frequencies_hz": [3.0],
                "violation_plot_input_paths": [],
                "save_filtered_csv": True,
                "allow_external_input": False,
                "force": False,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(repository)
    exit_code = main(["extrema", "run", "--config", "outputs/filter-request.json"])

    run_dir = repository / "outputs" / "angle_extrema" / "runs" / "causal-filter"
    filtered_path = (
        run_dir
        / "sources"
        / "person"
        / "session"
        / "stepintegrated_angle"
        / "filtered"
        / "lowpass_3hz.csv"
    )
    with filtered_path.open(encoding="utf-8", newline="") as csv_file:
        filtered = [[float(value) for value in row] for row in csv.reader(csv_file)]
    results = json.loads((run_dir / "extrema_results.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert [item["signal_variant"] for item in results["summaries"]] == [
        "raw",
        "lowpass_3hz",
    ]
    assert filtered[99][0] == pytest.approx(0.0, abs=1e-15)
    assert 0.0 < filtered[100][0] < 1.0
    assert [row[1] for row in filtered] == pytest.approx([2.0] * 201)
    assert [row[3] for row in filtered] == pytest.approx(
        [index * 0.01 for index in range(201)]
    )


def test_run_plots_every_violating_window_for_explicitly_selected_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _make_repository(tmp_path)
    relative_source = Path(
        "outputs/preprocessing/person/session/sampleintegrated_angle.csv"
    )
    source = repository / relative_source
    source.parent.mkdir(parents=True)
    with source.open("w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file).writerows(
            [
                [0.0, 0.0, 0.0, 0.00],
                [0.5, 0.0, 0.0, 0.25],
                [1.2, 0.0, 0.0, 0.50],
                [-0.5, 0.0, 0.0, 0.75],
                [-0.6, 0.0, 0.0, 1.00],
                [0.6, 0.0, 0.0, 1.25],
                [0.5, 0.0, 0.0, 1.50],
            ]
        )
    request = repository / "outputs" / "plot-request.json"
    request.write_text(
        json.dumps(
            {
                "run_name": "plot-windows",
                "input_paths": [relative_source.as_posix()],
                "window_duration_s": 1.0,
                "stride_duration_s": 0.5,
                "lower_bound_deg": -1.0,
                "upper_bound_deg": 1.0,
                "cutoff_frequencies_hz": [],
                "violation_plot_input_paths": [relative_source.as_posix()],
                "save_filtered_csv": True,
                "allow_external_input": False,
                "force": False,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(repository)
    exit_code = main(["extrema", "run", "--config", "outputs/plot-request.json"])

    plot_dir = (
        repository
        / "outputs"
        / "angle_extrema"
        / "runs"
        / "plot-windows"
        / "sources"
        / "person"
        / "session"
        / "sampleintegrated_angle"
        / "violation_windows"
        / "raw"
    )
    plots = sorted(plot_dir.glob("*.png"))

    assert exit_code == 0
    assert [plot.name for plot in plots] == [
        "window_000000_start_0s.png",
        "window_000001_start_0p5s.png",
    ]
    assert all(plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for plot in plots)


def test_named_run_arguments_save_the_same_complete_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _make_repository(tmp_path)
    relative = "outputs/preprocessing/session/sampleintegrated_angle.csv"
    _write_angle_file(repository / relative, 3, dt=0.5)

    monkeypatch.chdir(repository)
    exit_code = main(
        [
            "extrema",
            "run",
            "--input-paths",
            relative,
            "--run-name",
            "named-mode",
            "--window-duration-s",
            "1",
            "--stride-duration-s",
            "0.5",
            "--lower-bound-deg",
            "-1",
            "--upper-bound-deg",
            "1",
            "--cutoff-frequencies-hz",
        ]
    )

    saved = json.loads(
        (
            repository / "outputs/angle_extrema/runs/named-mode/analysis_config.json"
        ).read_text(encoding="utf-8")
    )
    timeline = (
        repository
        / "outputs"
        / "angle_extrema"
        / "runs"
        / "named-mode"
        / "sources"
        / "session"
        / "sampleintegrated_angle"
        / "raw_violation_timeline.png"
    )
    assert exit_code == 0
    assert saved == {
        "run_name": "named-mode",
        "input_paths": [relative],
        "window_duration_s": 1.0,
        "stride_duration_s": 0.5,
        "lower_bound_deg": -1.0,
        "upper_bound_deg": 1.0,
        "cutoff_frequencies_hz": [],
        "violation_plot_input_paths": [],
        "save_filtered_csv": True,
        "allow_external_input": False,
        "force": False,
    }
    assert timeline.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (["not", "an", "object"], "one object"),
        ({"input_paths": ["x"]}, "run_name"),
        ({"run_name": "x", "input_paths": "x"}, "non-empty array"),
        ({"run_name": "x", "input_paths": ["x"], "extra": True}, "unknown"),
        (
            {"run_name": "x", "input_paths": ["x"], "force": "yes"},
            "boolean",
        ),
    ],
)
def test_run_rejects_invalid_json_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: object,
    message: str,
) -> None:
    repository = _make_repository(tmp_path)
    request = repository / "outputs" / "invalid-run.json"
    request.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(repository)

    exit_code = main(["extrema", "run", "--config", "outputs/invalid-run.json"])

    assert exit_code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"output": "x"}, "input_root"),
        (
            {
                "input_root": "outputs/preprocessing",
                "output": "outputs/angle_extrema/selections/x.json",
                "quantile": "middle",
            },
            "number",
        ),
        (
            {
                "input_root": "outputs/preprocessing",
                "output": "outputs/angle_extrema/selections/x.json",
                "unexpected": True,
            },
            "unknown",
        ),
    ],
)
def test_select_rejects_invalid_json_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: object,
    message: str,
) -> None:
    repository = _make_repository(tmp_path)
    request = repository / "outputs" / "invalid-select.json"
    request.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(repository)

    exit_code = main(["extrema", "select", "--config", "outputs/invalid-select.json"])

    assert exit_code == 2
    assert message in capsys.readouterr().err


def test_extrema_help_documents_commands_and_run_parameters(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as group_exit:
        main(["extrema", "--help"])
    group_help = capsys.readouterr().out

    with pytest.raises(SystemExit) as run_exit:
        main(["extrema", "run", "--help"])
    run_help = capsys.readouterr().out

    assert group_exit.value.code == 0
    assert run_exit.value.code == 0
    assert "select" in group_help
    assert "run" in group_help
    assert "--input-paths" in run_help
    assert "--cutoff-frequencies-hz" in run_help
    assert "--violation-plot-input-paths" in run_help
    assert "--allow-external-input" in run_help
