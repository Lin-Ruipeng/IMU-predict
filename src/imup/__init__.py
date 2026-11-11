from collections.abc import Sequence

from imup.cli import main as _cli_main


def main(argv: Sequence[str] | None = None) -> int:
    """Run the IMU Predict command-line interface."""
    return _cli_main(argv)
