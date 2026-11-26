from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import signal  # type: ignore[import-untyped]

from imup.angle_extrema.core import AngleExtremaError

_ELLIPTIC_ORDER = 7
_PASSBAND_RIPPLE_DB = 0.1
_STOPBAND_ATTENUATION_DB = 60.0
_STOPBAND_EDGE_RATIO = 15 / 11
FILTER_FAMILIES = ("butterworth", "elliptic")
DEFAULT_FILTER_FAMILY = "butterworth"

FloatArray = npt.NDArray[np.float64]


def read_filter_family(value: object, *, path: Path | None = None) -> str:
    if not isinstance(value, str) or value not in FILTER_FAMILIES:
        raise AngleExtremaError(
            "filter_family must be 'butterworth' or 'elliptic'", path=path
        )
    return value


def filter_specification(
    family: str, cutoffs_hz: tuple[float, ...]
) -> dict[str, object]:
    if family == "elliptic":
        return {
            "family": family,
            "order": _ELLIPTIC_ORDER,
            "filtered_signal": "angular_velocity",
            "passband_ripple_db": _PASSBAND_RIPPLE_DB,
            "stopband_attenuation_db": _STOPBAND_ATTENUATION_DB,
            "passband_edges_hz": list(cutoffs_hz),
            "stopband_edges_hz": [
                cutoff * _STOPBAND_EDGE_RATIO for cutoff in cutoffs_hz
            ],
        }
    return {
        "family": family,
        "order": 4,
        "filtered_signal": "integrated_angle",
        "cutoffs_hz": list(cutoffs_hz),
    }


def comparison_reference_label(family: str, cutoff_hz: float) -> str:
    if family == "elliptic":
        return f"Elliptic reference ({cutoff_hz:g} Hz)"
    return f"Target Integrated Angle ({cutoff_hz:g} Hz)"


def filter_angle_rows(
    rows: FloatArray, sample_interval_s: float, cutoff_hz: float, family: str
) -> FloatArray:
    sampling_hz = 1.0 / sample_interval_s
    nyquist_hz = sampling_hz / 2
    if cutoff_hz >= nyquist_hz:
        raise AngleExtremaError(
            f"cutoff must be below Nyquist frequency {nyquist_hz:.12g} Hz"
        )

    if family == "elliptic":
        stopband_hz = cutoff_hz * _STOPBAND_EDGE_RATIO
        if stopband_hz >= nyquist_hz:
            raise AngleExtremaError(
                f"stopband edge {stopband_hz:.12g} Hz must be below Nyquist "
                f"frequency {nyquist_hz:.12g} Hz"
            )
        minimum_order, _ = signal.ellipord(
            cutoff_hz,
            stopband_hz,
            _PASSBAND_RIPPLE_DB,
            _STOPBAND_ATTENUATION_DB,
            fs=sampling_hz,
        )
        if minimum_order > _ELLIPTIC_ORDER:
            raise AngleExtremaError(
                f"elliptic specification requires order {minimum_order}, "
                f"above the configured order {_ELLIPTIC_ORDER}"
            )
        sections = signal.ellip(
            _ELLIPTIC_ORDER,
            _PASSBAND_RIPPLE_DB,
            _STOPBAND_ATTENUATION_DB,
            cutoff_hz,
            btype="lowpass",
            fs=sampling_hz,
            output="sos",
        )
        # Preprocessing integrates one fixed-step angular velocity per increment.
        angular_velocity = np.diff(rows[:, :3], axis=0) / sample_interval_s
        filtered_rows = rows.copy()
        filtered_rows[1:, :3] = rows[0, :3] + np.cumsum(
            _filter_axes(angular_velocity, sections) * sample_interval_s, axis=0
        )
        return filtered_rows

    sections = signal.butter(
        4, cutoff_hz, btype="lowpass", fs=sampling_hz, output="sos"
    )
    filtered_rows = rows.copy()
    filtered_rows[:, :3] = _filter_axes(rows[:, :3], sections)
    return filtered_rows


def _filter_axes(data: FloatArray, sections: FloatArray) -> FloatArray:
    steady_state = signal.sosfilt_zi(sections)
    filtered = np.empty_like(data)
    for index in range(3):
        filtered[:, index], _ = signal.sosfilt(
            sections, data[:, index], zi=steady_state * data[0, index]
        )
    return filtered
