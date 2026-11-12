from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from imup.angle_extrema import (
    AnalysisConfig,
    AngleExtremaError,
    ComparisonConfig,
    SelectConfig,
    load_analysis_config,
    load_comparison_config,
    load_select_config,
    run_angle_comparison,
    run_extrema_analysis,
    select_representative_files,
)
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

    extrema_parser = subparsers.add_parser(
        "extrema",
        help="select and analyze angle-extrema experiment inputs",
    )
    extrema_subparsers = extrema_parser.add_subparsers(
        dest="extrema_command",
        required=True,
    )
    select_parser = extrema_subparsers.add_parser(
        "select",
        help="select one line-count quantile file per directory",
    )
    select_parser.add_argument(
        "--config",
        type=Path,
        help="complete JSON selection configuration",
    )
    select_parser.add_argument(
        "--input-root",
        type=Path,
        help="root searched recursively for integrated-angle files",
    )
    select_parser.add_argument(
        "--quantile",
        type=float,
        default=None,
        help="nearest-rank line-count quantile to select (default: 0.5)",
    )
    select_parser.add_argument(
        "--output",
        type=Path,
        help="JSON report path",
    )
    select_parser.add_argument(
        "--force",
        action="store_true",
        default=None,
        help="replace an existing selection report",
    )

    run_parser = extrema_subparsers.add_parser(
        "run",
        help="run an angle-extrema experiment",
    )
    run_source = run_parser.add_mutually_exclusive_group(required=True)
    run_source.add_argument(
        "--config",
        type=Path,
        help="complete JSON experiment configuration",
    )
    run_source.add_argument(
        "--input-paths",
        nargs="+",
        type=Path,
        help="integrated-angle files or direct-child directories",
    )
    run_parser.add_argument("--run-name", help="ASCII experiment identifier")
    run_parser.add_argument(
        "--window-duration-s",
        type=float,
        help="physical window duration in seconds (default: 1.0)",
    )
    run_parser.add_argument(
        "--stride-duration-s",
        type=float,
        help="physical stride duration in seconds (default: 0.5)",
    )
    run_parser.add_argument(
        "--lower-bound-deg",
        type=float,
        help="strict lower bound for window-relative angles (default: -1)",
    )
    run_parser.add_argument(
        "--upper-bound-deg",
        type=float,
        help="strict upper bound for window-relative angles (default: 1)",
    )
    run_parser.add_argument(
        "--cutoff-frequencies-hz",
        nargs="*",
        type=float,
        help="ordered causal low-pass cutoffs; pass no values for raw only",
    )
    run_parser.add_argument(
        "--violation-plot-input-paths",
        nargs="*",
        type=Path,
        help="analyzed source files whose violating windows should be plotted",
    )
    run_parser.add_argument(
        "--save-filtered-csv",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="save each causal filtered signal as CSV (default: true)",
    )
    run_parser.add_argument(
        "--allow-external-input",
        action="store_true",
        default=None,
        help="allow inputs outside outputs/preprocessing",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        default=None,
        help="atomically replace an existing run with the same name",
    )

    compare_parser = extrema_subparsers.add_parser(
        "compare",
        help="compare angle variants with the target integrated angle",
    )
    compare_source = compare_parser.add_mutually_exclusive_group(required=True)
    compare_source.add_argument(
        "--config",
        type=Path,
        help="complete JSON comparison configuration",
    )
    compare_source.add_argument(
        "--analysis-run-name",
        help="existing angle-extrema run to compare",
    )
    compare_parser.add_argument(
        "--comparison-name",
        help="ASCII comparison output identifier",
    )
    compare_parser.add_argument(
        "--target-cutoff-hz",
        type=float,
        help="target integrated-angle cutoff (default: 11)",
    )
    compare_parser.add_argument(
        "--comparison-cutoff-frequencies-hz",
        nargs="*",
        type=float,
        help="filtered variants compared with the target (default: 3 5 7 9)",
    )
    compare_parser.add_argument(
        "--include-raw",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="compare raw integrated angle with the target (default: true)",
    )
    compare_parser.add_argument(
        "--force",
        action="store_true",
        default=None,
        help="atomically replace an existing comparison output",
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


def _run_extrema_select(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    repository_root = find_repository_root()
    if args.config is not None:
        if any(
            value is not None
            for value in (args.input_root, args.quantile, args.output, args.force)
        ):
            parser.error("--config cannot be combined with selection overrides")
        config = load_select_config(args.config, repository_root=repository_root)
    else:
        if args.input_root is None or args.output is None:
            parser.error("--input-root and --output are required without --config")
        config = SelectConfig(
            input_root=args.input_root,
            output=args.output,
            quantile=0.5 if args.quantile is None else float(args.quantile),
            force=False if args.force is None else bool(args.force),
        )
    result = select_representative_files(
        config,
        repository_root=repository_root,
    )
    for warning in result.warnings:
        _print_diagnostic("WARNING", warning.path, warning.message)
    for detail in result.details:
        _print_diagnostic(
            "INFO",
            detail.selected_path,
            (
                f"selected quantile={detail.quantile:g}; rows={detail.row_count}; "
                f"directory_files={detail.directory_file_count}"
            ),
        )
    _print_diagnostic("INFO", result.output_path, "saved selection report")
    print(
        "SUMMARY: "
        f"directories={len(result.details)} "
        f"selected={len(result.details)} "
        f"warnings={len(result.warnings)}",
        file=sys.stderr,
    )
    return 0


def _run_extrema_analysis(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    repository_root = find_repository_root()
    config_path: Path | None = args.config
    override_names = (
        "run_name",
        "window_duration_s",
        "stride_duration_s",
        "lower_bound_deg",
        "upper_bound_deg",
        "cutoff_frequencies_hz",
        "violation_plot_input_paths",
        "save_filtered_csv",
        "allow_external_input",
        "force",
    )
    if config_path is not None:
        if any(getattr(args, name) is not None for name in override_names):
            parser.error("--config cannot be combined with run parameter overrides")
        config = load_analysis_config(config_path, repository_root=repository_root)
    else:
        if args.run_name is None:
            parser.error("--run-name is required when --config is not used")
        config = AnalysisConfig(
            run_name=str(args.run_name),
            input_paths=tuple(args.input_paths),
            window_duration_s=(
                1.0 if args.window_duration_s is None else float(args.window_duration_s)
            ),
            stride_duration_s=(
                0.5 if args.stride_duration_s is None else float(args.stride_duration_s)
            ),
            lower_bound_deg=(
                -1.0 if args.lower_bound_deg is None else float(args.lower_bound_deg)
            ),
            upper_bound_deg=(
                1.0 if args.upper_bound_deg is None else float(args.upper_bound_deg)
            ),
            cutoff_frequencies_hz=(
                (11.0, 9.0, 7.0, 5.0, 3.0)
                if args.cutoff_frequencies_hz is None
                else tuple(args.cutoff_frequencies_hz)
            ),
            violation_plot_input_paths=(
                ()
                if args.violation_plot_input_paths is None
                else tuple(args.violation_plot_input_paths)
            ),
            save_filtered_csv=(
                True if args.save_filtered_csv is None else args.save_filtered_csv
            ),
            allow_external_input=(
                False
                if args.allow_external_input is None
                else args.allow_external_input
            ),
            force=False if args.force is None else args.force,
        )

    result = run_extrema_analysis(config, repository_root=repository_root)
    for source in result.processed:
        _print_diagnostic("INFO", source, "completed angle-extrema analysis")
    for warning in result.warnings:
        _print_diagnostic("WARNING", warning.path, warning.message)
    for failure in result.failures:
        _print_diagnostic("ERROR", failure.path, failure.message)
    _print_diagnostic("INFO", result.run_dir, "published angle-extrema run")
    print(
        "SUMMARY: "
        f"processed={len(result.processed)} "
        f"failed={len(result.failures)} "
        f"warnings={len(result.warnings)}",
        file=sys.stderr,
    )
    return result.exit_code


def _run_extrema_comparison(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    repository_root = find_repository_root()
    config_path: Path | None = args.config
    override_names = (
        "analysis_run_name",
        "comparison_name",
        "target_cutoff_hz",
        "comparison_cutoff_frequencies_hz",
        "include_raw",
        "force",
    )
    if config_path is not None:
        if any(getattr(args, name) is not None for name in override_names):
            parser.error("--config cannot be combined with comparison overrides")
        config = load_comparison_config(config_path, repository_root=repository_root)
    else:
        if args.analysis_run_name is None or args.comparison_name is None:
            parser.error(
                "--analysis-run-name and --comparison-name are required without --config"
            )
        config = ComparisonConfig(
            comparison_name=str(args.comparison_name),
            analysis_run_name=str(args.analysis_run_name),
            target_cutoff_hz=(
                11.0 if args.target_cutoff_hz is None else float(args.target_cutoff_hz)
            ),
            comparison_cutoff_frequencies_hz=(
                (3.0, 5.0, 7.0, 9.0)
                if args.comparison_cutoff_frequencies_hz is None
                else tuple(args.comparison_cutoff_frequencies_hz)
            ),
            include_raw=True if args.include_raw is None else bool(args.include_raw),
            force=False if args.force is None else bool(args.force),
        )
    result = run_angle_comparison(config, repository_root=repository_root)
    for warning in result.warnings:
        _print_diagnostic("WARNING", warning.path, warning.message)
    for failure in result.failures:
        _print_diagnostic("ERROR", failure.path, failure.message)
    _print_diagnostic("INFO", result.output_dir, "published angle comparison")
    print(
        "SUMMARY: "
        f"processed={result.processed_sources} "
        f"comparisons={result.comparisons} "
        f"failed={len(result.failures)} "
        f"warnings={len(result.warnings)}",
        file=sys.stderr,
    )
    return result.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "preprocess":
            return _run_preprocess(args, parser)
        if args.command == "extrema" and args.extrema_command == "select":
            return _run_extrema_select(args, parser)
        if args.command == "extrema" and args.extrema_command == "run":
            return _run_extrema_analysis(args, parser)
        if args.command == "extrema" and args.extrema_command == "compare":
            return _run_extrema_comparison(args, parser)
    except (PreprocessingError, AngleExtremaError) as error:
        _print_diagnostic("ERROR", error.path, str(error))
        return 2

    parser.error(f"unsupported command: {args.command}")
