# Repository guidelines

## Project structure

- Production code lives under `src/imup/<feature>/`.
- Tests mirror features under `tests/<feature>/`.
- Feature documentation lives under `docs/<feature>/`.
- `data/` and `outputs/` are local-only and must not be committed.
- Domain terms and design decisions are managed lazily by `grill-with-docs`.

## Development workflow

- Use Python 3.12 and manage dependencies and commands with uv.
- Use the `cu130` extra for local NVIDIA development and `cpu` in CI; never
  enable both extras together.
- Run pre-commit on staged files before committing. It runs Ruff and Mypy, not
  tests.
- CI is the final gate for the full test suite, Ruff, and Mypy.

## Observable operations

- Surface every detected anomaly. Data-processing CLIs write `INFO`, `WARNING`,
  or `ERROR` messages to stderr with the affected path and reason, followed by
  a batch summary; reusable library code returns diagnostics or raises errors.
- Require an explicit option before correcting names, applying fallbacks, or
  replacing generated output.

## Experiment CLI configuration

- Every parameterized data-processing command supports either complete named
  CLI arguments or a JSON object whose keys match those arguments. The two
  input modes are mutually exclusive.
- Resolve JSON paths from the repository root, reject unknown keys and invalid
  types, and persist the normalized arguments beside generated output. A
  command without generated output is exempt only from the persistence step.

## Agent skills

### Issue tracker

Issues are tracked in `Lin-Ruipeng/IMU-predict` GitHub Issues using the `gh`
CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The tracker uses the canonical triage labels. See
`docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository. Domain terms live in `GLOSSARY.md` and
decisions in `docs/adr/`, created lazily by `grill-with-docs`. See
`docs/agents/domain.md`.
