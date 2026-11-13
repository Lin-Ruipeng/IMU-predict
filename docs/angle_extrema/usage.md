# Angle-extrema analysis

The `angle_extrema` module finds integrated-angle windows whose window-relative
angle crosses configured bounds. It consumes the headerless
`x_deg,y_deg,z_deg,elapsed_s` files produced by preprocessing; it never
reintegrates angular velocity. See [`GLOSSARY.md`](../../GLOSSARY.md) for the
domain terms and [`docs/windowing.md`](../windowing.md) for the project-wide
window discretization contract.

Use `uv run imup extrema --help`, `uv run imup extrema select --help`,
`uv run imup extrema run --help`, `uv run imup extrema compare --help`, and
`uv run imup extrema compare-violations --help` for the authoritative command
reference.

## Select representative inputs

The selection utility recursively groups integrated-angle files by their direct
parent directory, sorts each group by `(physical row count, relative path)`, and
selects one file with the nearest-rank rule:

```text
index = ceil(quantile * file_count) - 1
```

The current experiment uses the median file from each directory:

```bash
uv run imup extrema select \
  --input-root outputs/preprocessing \
  --quantile 0.5 \
  --output outputs/angle_extrema/selections/q50.json
```

The report contains a copy-ready `selected_input_paths` array plus detailed row
counts and the normalized selection configuration. It is an inspection report,
not a special input to `extrema run`; copy the selected paths into the formal
run configuration. Selection is based only on row count and does not silently
exclude or replace files with known sampling gaps. Fewer than three files in a
directory produces a warning but still selects the requested actual file.

The equivalent selection JSON is:

```json
{
  "input_root": "outputs/preprocessing",
  "quantile": 0.5,
  "output": "outputs/angle_extrema/selections/q50.json",
  "force": false
}
```

Run it with `uv run imup extrema select --config <path>`. JSON keys are strict,
and `force: true` is required to replace an existing report.

## Run an experiment

A formal run configuration contains every effective parameter:

```json
{
  "run_name": "median-q50-default-bounds",
  "input_paths": [
    "outputs/preprocessing/<directory>/<file>integrated_angle.csv"
  ],
  "window_duration_s": 1.0,
  "stride_duration_s": 0.5,
  "lower_bound_deg": -1.0,
  "upper_bound_deg": 1.0,
  "cutoff_frequencies_hz": [11.0, 9.0, 7.0, 5.0, 3.0],
  "violation_plot_input_paths": [],
  "save_filtered_csv": true,
  "allow_external_input": false,
  "force": false
}
```

```bash
uv run imup extrema run --config outputs/angle_extrema/run-config.json
```

`run_name` must start with an ASCII letter or number and may then contain ASCII
letters, numbers, `.`, `_`, and `-`.

`input_paths` accepts individual files and directories. A directory contributes
only its direct `*integrated_angle.csv` children; analysis never recursively
expands it. Repeating the same resolved file, including through overlapping
file and directory inputs, is a configuration error.

The command also supports the same complete set of named arguments:

```bash
uv run imup extrema run \
  --run-name median-q50-default-bounds \
  --input-paths "outputs/preprocessing/<directory>/<file>integrated_angle.csv" \
  --window-duration-s 1 \
  --stride-duration-s 0.5 \
  --lower-bound-deg -1 \
  --upper-bound-deg 1 \
  --cutoff-frequencies-hz 11 9 7 5 3
```

Named arguments and `--config` are mutually exclusive. Paths in JSON resolve
from the repository root. Inputs outside `outputs/preprocessing/` are reported
and skipped unless `allow_external_input` or `--allow-external-input` is set;
the override permits location only and never bypasses data validation.

## Window and extrema semantics

For each file, recover fixed `dt` from its synthetic `elapsed_s` axis and use:

```text
window_samples = ceil(window_duration_s / dt)
stride_samples = ceil(stride_duration_s / dt)
```

All windows in the file have identical point counts. The incomplete tail is
discarded, and a file shorter than one window succeeds with a visible warning.
Within every complete window, subtract each axis's first value. Values strictly
below `lower_bound_deg` or strictly above `upper_bound_deg` violate the range;
values equal to a bound do not.

The input must contain at least two rows, four finite numeric columns, an
`elapsed_s` axis starting at zero, and strictly increasing fixed intervals
within floating-point tolerance. The command does not repair or resample an
invalid input.

## Causal filtering

Each configured cutoff creates a fourth-order Butterworth low-pass filter in
second-order-section form. `scipy.signal.sosfilt` processes the complete signal
before windowing, independently for each axis, with a first-sample steady-state
initial condition. The operation is causal: it uses no future samples, applies
no zero-phase correction, and retains the real filter delay.

A cutoff must be positive, unique, and lower than the file's Nyquist frequency.
An invalid file/cutoff variant is recorded as a failure while other files and
valid variants continue. An empty cutoff array requests raw-only analysis.

The 11 Hz, fourth-order, causal Butterworth result is the **Target Integrated
Angle**, the model's only prediction target. Raw integrated angle and all other
cutoffs are diagnostic variants. See
[`filter-delay-report.md`](filter-delay-report.md) for measured delay ranges and
the estimated alignment contract.

## Compare angle variants with the target

`extrema compare` consumes the saved CSV variants from an existing analysis
run. It does not refilter the data. A complete JSON request is:

```json
{
  "comparison_name": "q50-target-angle",
  "analysis_run_name": "q50-default-bounds",
  "target_cutoff_hz": 11.0,
  "comparison_cutoff_frequencies_hz": [3.0, 5.0, 7.0, 9.0],
  "include_raw": true,
  "error_ecdf": false,
  "error_ecdf_percentile_guides": false,
  "force": false
}
```

```bash
uv run imup extrema compare --config <path>
```

The equivalent complete named-argument mode is:

```bash
uv run imup extrema compare \
  --comparison-name q50-target-angle \
  --analysis-run-name q50-default-bounds \
  --target-cutoff-hz 11 \
  --comparison-cutoff-frequencies-hz 3 5 7 9 \
  --include-raw
```

Named arguments and `--config` are mutually exclusive. Each source gets five
uncompensated and five estimated-delay-compensated plots. Every image has three
vertically stacked x/y/z panels: the Target Integrated Angle is solid, the
comparison is long-dashed, and a translucent band exposes small differences.
The compensated image discards non-overlapping data at both ends. Its title
does not disclose the estimated shift.

### Anchor-relative absolute-error ECDF

Pass `--error-ecdf`, or set `"error_ecdf": true`, to generate one absolute-error
ECDF image and one pointwise-error CSV for every successfully compared source.
The implementation iterates over all sources in the analysis run; it does not
assume a fixed source count. Add `--error-ecdf-percentile-guides`, or set
`"error_ecdf_percentile_guides": true`, to draw horizontal p50, p90, p95, and
p99 probability guides. The guide option requires error ECDF output to be
enabled. For example:

```bash
uv run imup extrema compare \
  --comparison-name q50-target-angle-error-ecdf \
  --analysis-run-name q50-default-bounds \
  --target-cutoff-hz 11 \
  --comparison-cutoff-frequencies-hz 3 5 7 9 \
  --include-raw \
  --error-ecdf \
  --error-ecdf-percentile-guides
```

The 11 Hz value is the Target Integrated Angle's low-pass cutoff, not a sample
rate. Let its unchanged time grid be `t_i`, let the signed estimated comparator
shift be `s`, and let the comparator samples be `C_a(u_j)` for axis
`a in {x, y, z}`. Retain the strict intersection

```text
I = {i | u_0 <= t_i - s <= u_m}
```

and query the comparator at `q_i = t_i - s`. If `q_i` equals a comparator
sample time, use that sample directly. Otherwise choose adjacent times with
`u_j < q_i < u_(j+1)` and compute

```text
alpha_i = (q_i - u_j) / (u_(j+1) - u_j)
C_tilde_a(t_i) = (1 - alpha_i) C_a(u_j) + alpha_i C_a(u_(j+1))
```

This linear interpolation produces exactly one comparator value for every
retained target time; it does not add target time points, extrapolate, or wrap
either edge. Let `i_0 = min(I)` be the first retained target point. Independently
form the two anchor-relative angles

```text
T_prime_a(t_i) = T_a(t_i) - T_a(t_(i_0))
C_prime_a(t_i) = C_tilde_a(t_i) - C_tilde_a(t_(i_0))
```

and their pointwise absolute error

```text
e_(a,i) = abs(T_prime_a(t_i) - C_prime_a(t_i)).
```

For the `n` retained errors of one source, comparator, and axis, the plotted
empirical cumulative distribution is

```text
F_a(e) = count(e_(a,i) <= e) / n.
```

Its reported quantile is the observed inverse-ECDF value
`Q_a(p) = inf {e | F_a(e) >= p}`: after sorting, take observation `ceil(n p)`.
No additional interpolation between error values is applied. Each image has
independently scaled x/y/z panels; every panel overlays raw and the configured
cutoff variants as step curves on one absolute-error axis, with ECDF fixed to
`[0, 1]`.

Comparison output is atomically published under:

```text
outputs/angle_extrema/comparisons/<comparison_name>/
├── comparison_config.json
├── comparison_results.json
├── delay_compensation.csv
├── absolute_error_summary.csv       # only with error_ecdf
├── README.md
└── sources/
    └── <preprocessing-relative-source>/
        ├── absolute_error_ecdf.png   # only with error_ecdf
        ├── absolute_errors.csv       # only with error_ecdf
        ├── raw_vs_target_11hz_without_delay_compensation.png
        ├── raw_vs_target_11hz_with_delay_compensation.png
        └── ...
```

`delay_compensation.csv` records the shared tri-axis shift, direction,
correlation, and retained time range for every source/variant pair. Existing
output requires explicit `force: true`; replacement is run-wide.
`absolute_errors.csv` records `target_elapsed_s`, the comparator machine name,
and x/y/z absolute errors for every retained point. Because each comparator can
have a different strict intersection, its rows remain a separate long-table
group. `absolute_error_summary.csv` contains one row per source, comparator,
and axis with the retained sample count, p50, p90, p95, p99, and maximum error.

## Compare Target Violation Windows

`extrema compare-violations` consumes both an existing angle-extrema run and
its existing target-angle comparison. It selects only records whose signal
variant is the 11 Hz Target Integrated Angle; violations detected in raw or
other cutoff variants never select a window. The command reuses each source and
comparator's full-series shared tri-axis shift from `delay_compensation.csv` and
does not refilter or estimate a shift per window.

A complete JSON request is:

```json
{
  "comparison_name": "q50-target-angle",
  "analysis_run_name": "q50-default-bounds",
  "force": false
}
```

```bash
uv run imup extrema compare-violations --config <path>
```

The equivalent complete named-argument mode is:

```bash
uv run imup extrema compare-violations \
  --comparison-name q50-target-angle \
  --analysis-run-name q50-default-bounds
```

Every Target Violation Window produces one pairwise image for raw and each
configured diagnostic cutoff. The target remains on its original window time
grid. A comparator with signed shift `s` is queried at `target_time - s` by
linear sub-sample interpolation. Each curve independently subtracts its first
plotted three-axis sample, so complete-coverage curves all begin at `(0, 0)`.
Both curves are solid, use stable variant-specific colors, and enclose a
translucent difference band. Three vertically stacked x/y/z panels show both
configured bound lines, and panels whose target axis triggered selection are
marked `VIOLATION`. Figure titles disclose that delay compensation is applied
but not its magnitude.

At a source boundary, unavailable comparator data is not extrapolated or
wrapped. The complete target window remains visible, while the comparator
begins at its first available point and is rebased there. These images are
recorded as `PARTIAL_COVERAGE` warnings.

The command atomically publishes only this subtree and leaves the existing
full-series comparison files untouched:

```text
outputs/angle_extrema/comparisons/<comparison_name>/
└── target_violation_windows/
    ├── violation_comparison_config.json
    ├── violation_comparison_results.json
    ├── window_comparisons.csv
    ├── README.md
    └── sources/
        └── <preprocessing-relative-source>/
            ├── raw/
            │   └── window_<index>_start_<time>s.png
            ├── lowpass_3hz/
            └── ...
```

`window_comparisons.csv` records the target window, violated axes, full-series
shift, signed query interval, coverage status, and image path. Existing output
requires explicit `force: true`; replacement affects only
`target_violation_windows/`.

## Outputs

Each run is published under:

```text
outputs/angle_extrema/runs/<run_name>/
├── analysis_config.json
├── extrema_results.json
└── sources/
    └── <preprocessing-relative-source>/
        ├── raw_series.png
        ├── raw_violation_timeline.png
        ├── filtered/
        │   ├── lowpass_11hz.csv
        │   └── ...
        └── violation_windows/
            ├── raw/
            └── lowpass_11hz/
```

`raw_series.png` is the complete, unshifted integrated-angle signal: x/y/z share
one Seaborn-styled plot with fixed colors and no bound lines. Filtered CSV files
remain headerless `x_deg,y_deg,z_deg,elapsed_s` data. Filtered full-series plots
are not generated.

`raw_violation_timeline.png` preserves the unfiltered integrated-angle signal in
three vertically stacked x/y/z panels with a shared elapsed-time axis. Each
panel marks only that axis's raw violation time coverage at `y=0`. Overlapping
or touching half-open violation windows are merged before plotting, so every red
segment means that at least one raw violating window covers that time. The
timeline is generated for every successfully analyzed source and does not
replace `raw_series.png`.

`extrema_results.json` contains per-file/per-variant window summaries, one
record for every violating window, all three axes' minima and maxima, violated
axes/directions, failures, warnings, and a `succeeded` or `failed` run status.
By default no violation-window images are generated. Paths listed in
`violation_plot_input_paths` receive one image per violating raw or filtered
window; these images show window-relative x/y/z curves and both bound lines.

Run construction occurs in a temporary directory. A completed attempt is then
published as a whole, including a fully accounted `failed` attempt. An
interrupted attempt is not published. Existing runs require `force: true`; the
replacement is run-wide so removed cutoffs cannot leave stale artifacts.
