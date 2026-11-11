import csv
import json
from pathlib import Path

import pytest

from imup import main


def _make_cli_repository(tmp_path: Path) -> Path:
    (tmp_path / "src" / "imup").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    input_dir = tmp_path / "data" / "person" / "session"
    input_dir.mkdir(parents=True)
    with (input_dir / "VID_001gyro.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        csv.writer(csv_file).writerows(
            [
                ["discarded"],
                [1.0, 0.0, 0.0, 0],
                [1.0, 0.0, 0.0, 10_000_000],
            ]
        )
    return tmp_path


def test_cli_and_json_modes_produce_the_same_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _make_cli_repository(tmp_path)
    monkeypatch.chdir(repository)

    cli_exit = main(["preprocess", "--input-dir", "data/person/session"])
    first_stderr = capsys.readouterr().err
    output_path = (
        repository
        / "outputs"
        / "preprocessing"
        / "person"
        / "session"
        / "VID_001integrated_angle.csv"
    )
    cli_output = output_path.read_text(encoding="utf-8")

    config_path = repository / "outputs" / "request.json"
    config_path.write_text(
        json.dumps({"input_dir": "data/person/session", "force": True}),
        encoding="utf-8",
    )
    json_exit = main(["preprocess", "--config", "outputs/request.json"])
    second_stderr = capsys.readouterr().err

    assert cli_exit == 0
    assert json_exit == 0
    assert output_path.read_text(encoding="utf-8") == cli_output
    assert "frequency_hz=100.000000" in first_stderr
    assert "SUMMARY: processed=1" in first_stderr
    assert "SUMMARY: processed=1" in second_stderr


def test_config_mode_rejects_cli_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _make_cli_repository(tmp_path)
    monkeypatch.chdir(repository)
    config_path = repository / "outputs" / "request.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps({"input_dir": "data/person/session", "force": False}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as error:
        main(["preprocess", "--config", "outputs/request.json", "--force"])

    assert error.value.code == 2


def test_cli_prints_json_errors_and_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _make_cli_repository(tmp_path)
    monkeypatch.chdir(repository)
    config_path = repository / "outputs" / "request.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps({"input_dir": "data/person/session", "unexpected": True}),
        encoding="utf-8",
    )

    exit_code = main(["preprocess", "--config", "outputs/request.json"])
    stderr = capsys.readouterr().err

    assert exit_code == 2
    assert "ERROR:" in stderr
    assert "unknown preprocessing JSON keys" in stderr
