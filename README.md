# IMU Predict

A collection of experiments and tools for IMU data analysis and prediction.

## Setup

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/) for dependency
management.

For local development with NVIDIA CUDA:

```bash
uv sync --extra cu130 --dev
uv run pre-commit install
```

For a CPU-only environment, including CI:

```bash
uv sync --extra cpu --dev
```

The `cpu` and `cu130` extras are mutually exclusive.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest --cov=imup
uv run pre-commit run --all-files
```

Pre-commit runs Ruff and Mypy on staged Python files. The full test suite runs
in CI.

## Project layout

```text
src/imup/<feature>/   Production code grouped by feature
tests/<feature>/      Tests mirroring feature code
docs/<feature>/       Feature documentation
docs/agents/          Repository conventions consumed by engineering skills
docs/adr/             Design decisions, created lazily by grill-with-docs
data/                  Local datasets; ignored by Git
outputs/               Generated artifacts; ignored by Git
```

Keep reusable code under `src/imup/`. Avoid placing importable feature modules
at the repository root. `grill-with-docs` creates `GLOSSARY.md` and ADRs only
when the project has real terms or decisions to record.
