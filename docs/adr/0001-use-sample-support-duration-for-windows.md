---
status: accepted
---

# Use sample support duration for windows

Define a window or stride containing `N` samples as supporting `N × dt` seconds
and discretize a requested duration `D` as `ceil(D / dt)` samples. This keeps
every window in a file the same size and makes its supported duration no shorter
than requested; we deliberately reject the first-to-last-sample span
`(N - 1) × dt`, which would require an extra sample and conflict with the
project's 99 Hz and 98.1 Hz examples. The operational contract lives in
[`docs/windowing.md`](../windowing.md).
