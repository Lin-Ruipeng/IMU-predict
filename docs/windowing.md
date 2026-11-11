# Sliding-window contract

This document is the project-wide source of truth for converting physical
durations into sliding windows. Feature documentation may reference this
contract but must not redefine it.

## Time basis

Each input file has one fixed sample period `dt`; timestamp jitter has already
been removed before windowing. A window's supported duration is defined as its
sample count multiplied by `dt`, rather than the elapsed time between its first
and last samples.

For a requested duration `D`:

```text
sample_count = ceil(D / dt)
```

This is the smallest whole-sample window whose supported duration is not less
than `D`. Every window from the same file therefore contains the same number of
samples. For example, a one-second window contains 99 samples at both 99 Hz and
98.1 Hz.

## Stride and tail handling

Convert a requested stride `S` with the same rule:

```text
stride_samples = ceil(S / dt)
```

Window start rows are `0, stride_samples, 2 * stride_samples, ...`. Discard a
trailing segment that cannot fill one complete window. A file that cannot
produce any complete window remains valid but must produce a visible warning.

## Angle-signal consistency

Derive causal filtered variants from the complete angle sequence before
windowing. Apply identical window boundaries to every variant. Within each
window, subtract each axis's first value from that axis; this produces
window-relative angle and is not a new integration of angular velocity.

Record window positions using a zero-based start row and the corresponding
relative elapsed time. A window position is not a sensor timestamp.
