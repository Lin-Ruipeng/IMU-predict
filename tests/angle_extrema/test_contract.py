from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from imup import main


def _repository(tmp_path: Path) -> Path:
    (tmp_path / "src" / "imup").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "outputs" / "preprocessing").mkdir(parents=True)
    return tmp_path


def _write_rows(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows)


def _write_request(
    repository: Path,
    *,
    run_name: str,
    input_paths: list[str],
    force: bool = False,
    allow_external_input: bool = False,
    cutoffs: list[float] | None = None,
) -> Path:
    request = repository / "outputs" / f"{run_name}-request.json"
    request.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "input_paths": input_paths,
                "window_duration_s": 1.0,
                "stride_duration_s": 0.5,
                "lower_bound_deg": -1.0,
                "upper_bound_deg": 1.0,
                "cutoff_frequencies_hz": [] if cutoffs is None else cutoffs,
                "violation_plot_input_paths": [],
                "save_filtered_csv": True,
                "allow_external_input": allow_external_input,
                "force": force,
            }
        ),
        encoding="utf-8",
    )
    return request


def test_file_shorter_than_one_window_warns_without_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _repository(tmp_path)
    relative = "outputs/preprocessing/session/shortintegrated_angle.csv"
    _write_rows(
        repository / relative,
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.25],
            [0.2, 0.0, 0.0, 0.50],
        ],
    )
    request = _write_request(
        repository,
        run_name="short-file",
        input_paths=[relative],
    )

    monkeypatch.chdir(repository)
    exit_code = main(["extrema", "run", "--config", str(request)])

    results = json.loads(
        (
            repository
            / "outputs"
            / "angle_extrema"
            / "runs"
            / "short-file"
            / "extrema_results.json"
        ).read_text(encoding="utf-8")
    )
    stderr = capsys.readouterr().err

    assert exit_code == 0
    assert results["summaries"][0]["total_windows"] == 0
    assert results["warnings"][0]["path"] == relative
    assert "shorter than one complete window" in results["warnings"][0]["message"]
    assert "WARNING:" in stderr


def test_partial_failure_publishes_a_complete_failed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    good = "outputs/preprocessing/session/goodintegrated_angle.csv"
    bad = "outputs/preprocessing/session/badintegrated_angle.csv"
    _write_rows(
        repository / good,
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
            [2.0, 0.0, 0.0, 1.0],
        ],
    )
    _write_rows(
        repository / bad,
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.4],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    request = _write_request(
        repository,
        run_name="partial-failure",
        input_paths=[bad, good],
    )

    monkeypatch.chdir(repository)
    exit_code = main(["extrema", "run", "--config", str(request)])

    run_dir = repository / "outputs/angle_extrema/runs/partial-failure"
    results = json.loads((run_dir / "extrema_results.json").read_text(encoding="utf-8"))

    assert exit_code == 1
    assert results["run_status"] == "failed"
    assert [item["source_file"] for item in results["summaries"]] == [good]
    assert results["failures"][0]["path"] == bad
    assert "fixed" in results["failures"][0]["message"]
    assert (run_dir / "analysis_config.json").is_file()
    assert (run_dir / "sources/session/goodintegrated_angle/raw_series.png").is_file()


def test_invalid_cutoff_fails_only_that_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    relative = "outputs/preprocessing/session/slowintegrated_angle.csv"
    _write_rows(
        repository / relative,
        [
            [0.0, 0.0, 0.0, 0.00],
            [0.0, 0.0, 0.0, 0.25],
            [1.5, 0.0, 0.0, 0.50],
            [1.5, 0.0, 0.0, 0.75],
            [1.5, 0.0, 0.0, 1.00],
        ],
    )
    request = _write_request(
        repository,
        run_name="bad-cutoff",
        input_paths=[relative],
        cutoffs=[3.0, 1.0],
    )

    monkeypatch.chdir(repository)
    exit_code = main(["extrema", "run", "--config", str(request)])

    run_dir = repository / "outputs/angle_extrema/runs/bad-cutoff"
    results = json.loads((run_dir / "extrema_results.json").read_text(encoding="utf-8"))
    variants = [item["signal_variant"] for item in results["summaries"]]

    assert exit_code == 1
    assert variants == ["raw", "lowpass_1hz"]
    assert "lowpass_3hz" in results["failures"][0]["message"]
    assert "Nyquist" in results["failures"][0]["message"]
    assert (
        run_dir / "sources/session/slowintegrated_angle/filtered/lowpass_1hz.csv"
    ).is_file()


def test_external_input_requires_and_respects_an_explicit_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    external = repository / "external" / "sampleintegrated_angle.csv"
    _write_rows(
        external,
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    blocked = _write_request(
        repository,
        run_name="external-blocked",
        input_paths=["external/sampleintegrated_angle.csv"],
    )
    allowed = _write_request(
        repository,
        run_name="external-allowed",
        input_paths=["external/sampleintegrated_angle.csv"],
        allow_external_input=True,
    )

    monkeypatch.chdir(repository)
    blocked_exit = main(["extrema", "run", "--config", str(blocked)])
    allowed_exit = main(["extrema", "run", "--config", str(allowed)])

    blocked_results = json.loads(
        (
            repository
            / "outputs/angle_extrema/runs/external-blocked/extrema_results.json"
        ).read_text(encoding="utf-8")
    )
    allowed_dir = repository / "outputs/angle_extrema/runs/external-allowed"

    assert blocked_exit == 1
    assert blocked_results["summaries"] == []
    assert "allow_external_input" in blocked_results["warnings"][0]["message"]
    assert allowed_exit == 0
    assert len(list((allowed_dir / "sources/external").glob("*/raw_series.png"))) == 1


def test_duplicate_expanded_inputs_are_rejected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _repository(tmp_path)
    relative = "outputs/preprocessing/session/sampleintegrated_angle.csv"
    _write_rows(
        repository / relative,
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    )
    request = _write_request(
        repository,
        run_name="duplicates",
        input_paths=["outputs/preprocessing/session", relative],
    )

    monkeypatch.chdir(repository)
    exit_code = main(["extrema", "run", "--config", str(request)])

    assert exit_code == 2
    assert "duplicate files" in capsys.readouterr().err
    assert not (repository / "outputs/angle_extrema/runs/duplicates").exists()


def test_directory_inputs_scan_only_direct_angle_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    direct = "outputs/preprocessing/session/directintegrated_angle.csv"
    nested = "outputs/preprocessing/session/nested/hiddenintegrated_angle.csv"
    rows = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    _write_rows(repository / direct, rows)
    _write_rows(repository / nested, rows)
    request = _write_request(
        repository,
        run_name="direct-only",
        input_paths=["outputs/preprocessing/session"],
    )

    monkeypatch.chdir(repository)
    exit_code = main(["extrema", "run", "--config", str(request)])

    results = json.loads(
        (
            repository / "outputs/angle_extrema/runs/direct-only/extrema_results.json"
        ).read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert [item["source_file"] for item in results["summaries"]] == [direct]


def test_force_replaces_the_whole_run_without_stale_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _repository(tmp_path)
    relative = "outputs/preprocessing/session/sampleintegrated_angle.csv"
    _write_rows(
        repository / relative,
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    )
    first = _write_request(
        repository,
        run_name="replace-me",
        input_paths=[relative],
    )

    monkeypatch.chdir(repository)
    assert main(["extrema", "run", "--config", str(first)]) == 0
    run_dir = repository / "outputs/angle_extrema/runs/replace-me"
    stale = run_dir / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    assert main(["extrema", "run", "--config", str(first)]) == 2
    assert stale.is_file()
    capsys.readouterr()

    forced = _write_request(
        repository,
        run_name="replace-me",
        input_paths=[relative],
        force=True,
    )
    assert main(["extrema", "run", "--config", str(forced)]) == 0
    assert not stale.exists()
    assert (
        json.loads((run_dir / "analysis_config.json").read_text(encoding="utf-8"))[
            "force"
        ]
        is True
    )


def test_noninteger_frequency_uses_ceiling_width_and_fixed_ceiling_stride(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    relative = "outputs/preprocessing/session/fractionalintegrated_angle.csv"
    sample_interval = 1.0 / 98.1
    _write_rows(
        repository / relative,
        [[float(index), 0.0, 0.0, index * sample_interval] for index in range(199)],
    )
    request = _write_request(
        repository,
        run_name="fractional-frequency",
        input_paths=[relative],
    )

    monkeypatch.chdir(repository)
    assert main(["extrema", "run", "--config", str(request)]) == 0

    results = json.loads(
        (
            repository
            / "outputs/angle_extrema/runs/fractional-frequency/extrema_results.json"
        ).read_text(encoding="utf-8")
    )
    assert results["summaries"][0]["total_windows"] == 3
    assert [item["start_row"] for item in results["violations"]] == [0, 50, 100]
    assert results["violations"][0]["end_elapsed_s_exclusive"] == pytest.approx(
        99.0 / 98.1
    )


def test_values_equal_to_bounds_are_not_violations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    relative = "outputs/preprocessing/session/equalintegrated_angle.csv"
    _write_rows(
        repository / relative,
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    request = _write_request(
        repository,
        run_name="equal-bounds",
        input_paths=[relative],
    )

    monkeypatch.chdir(repository)
    assert main(["extrema", "run", "--config", str(request)]) == 0
    results = json.loads(
        (
            repository / "outputs/angle_extrema/runs/equal-bounds/extrema_results.json"
        ).read_text(encoding="utf-8")
    )
    assert results["violations"] == []


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([[0.0, 0.0, 0.0, 0.0]], "at least two"),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "expected exactly 4"),
        (
            [["bad", 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "non-numeric",
        ),
        (
            [[float("nan"), 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "non-finite",
        ),
        (
            [[0.0, 0.0, 0.0, 0.1], [0.0, 0.0, 0.0, 1.1]],
            "start at zero",
        ),
        (
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            "strictly increasing",
        ),
        (
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.4],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "fixed",
        ),
    ],
)
def test_invalid_angle_csv_is_reported_in_a_completed_failed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[list[object]],
    message: str,
) -> None:
    repository = _repository(tmp_path)
    relative = "outputs/preprocessing/session/invalidintegrated_angle.csv"
    _write_rows(repository / relative, rows)
    request = _write_request(
        repository,
        run_name="invalid-data",
        input_paths=[relative],
    )

    monkeypatch.chdir(repository)
    exit_code = main(["extrema", "run", "--config", str(request)])

    results = json.loads(
        (
            repository / "outputs/angle_extrema/runs/invalid-data/extrema_results.json"
        ).read_text(encoding="utf-8")
    )
    assert exit_code == 1
    assert results["run_status"] == "failed"
    assert message in results["failures"][0]["message"]
    assert results["summaries"] == []


def test_run_name_cannot_resolve_outside_the_runs_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _repository(tmp_path)
    relative = "outputs/preprocessing/session/sampleintegrated_angle.csv"
    _write_rows(
        repository / relative,
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    )
    request = _write_request(
        repository,
        run_name="..",
        input_paths=[relative],
        force=True,
    )

    monkeypatch.chdir(repository)
    exit_code = main(["extrema", "run", "--config", str(request)])

    assert exit_code == 2
    assert "must start" in capsys.readouterr().err
    assert (repository / "outputs" / "preprocessing").is_dir()
