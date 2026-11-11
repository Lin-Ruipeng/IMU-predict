from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from imup.preprocessing import (
    PreprocessConfig,
    PreprocessingError,
    find_repository_root,
    load_preprocess_config,
    preprocess_directory,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imup")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="integrate gyroscope recordings in one data directory",
    )
    source = preprocess_parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-dir",
        type=Path,
        help="direct child directory under data/ to process",
    )
    source.add_argument(
        "--config",
        type=Path,
        help="JSON file containing input_dir and force",
    )
    preprocess_parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing generated files",
    )
    return parser


def _print_diagnostic(level: str, path: Path | None, message: str) -> None:
    location = f" {path}:" if path is not None else ""
    print(f"{level}:{location} {message}", file=sys.stderr)


def _run_preprocess(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    repository_root = find_repository_root()

    config_path: Path | None = args.config
    if config_path is not None:
        if args.force:
            parser.error("--config cannot be combined with --force")
        config = load_preprocess_config(config_path, repository_root=repository_root)
    else:
        input_dir: Path | None = args.input_dir
        if input_dir is None:
            parser.error("--input-dir is required when --config is not used")
        config = PreprocessConfig(input_dir=input_dir, force=bool(args.force))

    result = preprocess_directory(config, repository_root=repository_root)

    for file_result in result.processed:
        _print_diagnostic(
            "INFO",
            file_result.source,
            (
                f"wrote {file_result.destination}; rows={file_result.output_rows}; "
                f"frequency_hz={file_result.estimated_frequency_hz:.6f}; "
                f"period_ms={file_result.sample_interval_seconds * 1_000:.6f}; "
                f"intervals_over_10pct={file_result.fraction_over_ten_percent:.6%}; "
                f"max_interval_ratio={file_result.max_interval_ratio:.6f}"
            ),
        )

    for diagnostic in result.warnings:
        _print_diagnostic("WARNING", diagnostic.path, diagnostic.message)
    for diagnostic in result.skipped:
        _print_diagnostic("WARNING", diagnostic.path, diagnostic.message)
    for diagnostic in result.failed:
        _print_diagnostic("ERROR", diagnostic.path, diagnostic.message)

    _print_diagnostic(
        "INFO",
        result.config_path,
        "saved normalized preprocessing configuration",
    )
    print(
        "SUMMARY: "
        f"processed={len(result.processed)} "
        f"ignored_accel={result.ignored_accel} "
        f"skipped={len(result.skipped)} "
        f"failed={len(result.failed)} "
        f"warnings={result.warning_count}",
        file=sys.stderr,
    )
    return result.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "preprocess":
            return _run_preprocess(args, parser)
    except PreprocessingError as error:
        _print_diagnostic("ERROR", error.path, str(error))
        return 2

    parser.error(f"unsupported command: {args.command}")
