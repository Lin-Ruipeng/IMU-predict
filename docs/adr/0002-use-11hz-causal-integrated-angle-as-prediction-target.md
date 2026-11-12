---
status: accepted
---

# Use 11 Hz causal integrated angle as the prediction target

Define the **Target Integrated Angle** as the raw angular velocity integrated
per axis and then processed by an 11 Hz, fourth-order, causal Butterworth
low-pass filter. It is the model's only prediction target and the reference for
angle-variant comparisons; the unfiltered integrated angle and other cutoff
variants are diagnostic signals only. Causal filtering keeps the target
constructible online while suppressing high-frequency integrated-angle
fluctuation, and changing the cutoff, order, filter family, or causality creates
a different prediction target rather than a compatible implementation detail.
