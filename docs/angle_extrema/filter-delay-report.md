# Causal Butterworth filter-delay report

This report records the expected delay scale of the fourth-order causal
Butterworth angle variants used in the 17-source comparison experiment. It
also explains why delay compensation is an estimated alignment rather than an
exact timestamp correction.

## Scope and method

The 17 sources have sampling frequencies from 206.980 to 499.961 Hz, with 15
distinct frequencies. For each source frequency and cutoff, the digital filter
was designed with the same `scipy.signal.butter(..., output="sos")` contract as
the analysis command. The second-order sections were converted to transfer
function form and evaluated with `scipy.signal.group_delay`.

Group delay is frequency-dependent:

```text
group_delay(f) = -d phase(f) / d angular_frequency
```

The low-frequency column below is the DC-limit delay and is useful only as a
delay-scale approximation. The cutoff column demonstrates how much the delay
changes within one filter. Values are ranges across the 17 sources' sampling
frequencies.

| Signal variant | Low-frequency delay | Delay near its own cutoff | Low-frequency delay relative to 11 Hz | Approximate alignment direction |
|---|---:|---:|---:|---|
| Raw integrated angle | 0 ms | Not applicable | 37.5–37.7 ms earlier | Delay raw by about 37.5–37.7 ms |
| 3 Hz low-pass | 138.5–138.6 ms | 196.1–196.3 ms | 100.9–101.1 ms later | Advance 3 Hz by about 101 ms |
| 5 Hz low-pass | 83.0–83.2 ms | 117.7–118.1 ms | 45.4–45.6 ms later | Advance 5 Hz by about 45.5 ms |
| 7 Hz low-pass | 59.2–59.4 ms | 84.1–84.7 ms | 21.6–21.7 ms later | Advance 7 Hz by about 21.7 ms |
| 9 Hz low-pass | 45.9–46.2 ms | 65.5–66.2 ms | 8.4–8.5 ms later | Advance 9 Hz by about 8.4 ms |
| 11 Hz Target Integrated Angle | 37.5–37.7 ms | 53.6–54.5 ms | Reference | Do not shift |

These theoretical directions are sanity checks, not the values applied to the
plots. A fourth-order Butterworth filter is an IIR system: each output combines
the current input with past state, its phase is nonlinear, and different
frequency components have different delays and amplitude changes. A complete
motion signal therefore cannot be converted into the filtered result by one
exact time translation. The steady-state initial condition reduces a constant
start transient but does not make the filter a pure delay.

## Observed best-fit shifts in the 17-source batch

The actual comparison run estimated the following shared tri-axis translations.
A positive raw shift means delaying raw; a filtered-variant value is shown as
the amount by which that variant is advanced. Correlation is the normalized
first-difference correlation at the selected shift.

| Comparator | Applied direction | Best-fit shift range | Mean shift | Correlation range |
|---|---|---:|---:|---:|
| Raw integrated angle | Delay | 39.4–43.3 ms | 41.6 ms | 0.9648–0.9992 |
| 3 Hz low-pass | Advance | 102.9–118.6 ms | 115.1 ms | 0.5818–0.9488 |
| 5 Hz low-pass | Advance | 50.5–55.2 ms | 53.5 ms | 0.7595–0.9881 |
| 7 Hz low-pass | Advance | 25.0–27.9 ms | 26.6 ms | 0.9044–0.9972 |
| 9 Hz low-pass | Advance | 9.6–11.1 ms | 10.6 ms | 0.9832–0.9996 |

All 85 estimates have the direction predicted by the theoretical group-delay
table. The lower 3 Hz correlation values are important rather than erroneous:
stronger smoothing changes waveform shape and amplitude, so even the best
single translation cannot make it equivalent to the 11 Hz target.

## Best-fit comparison alignment

The comparison command estimates one shared x/y/z translation independently
for every source and comparison variant:

1. Take the first difference of each axis to keep slow integrated drift from
   dominating the estimate.
2. Standardize every informative axis and combine their normalized
   cross-correlations.
3. Select the best global lag and refine it to sub-sample resolution around the
   correlation peak.
4. Keep the 11 Hz Target Integrated Angle fixed, linearly interpolate only the
   comparison variant, and discard both non-overlapping ends. Never fill,
   extrapolate, or wrap the signal.

The result is an **estimated best-fit shift** for shape comparison, not a claim
that every filtered point has that physical delay. Plot titles state only
whether estimated delay compensation is present. Exact shifts, directions,
correlations, and retained time ranges are recorded in the comparison output's
`delay_compensation.csv` file.
