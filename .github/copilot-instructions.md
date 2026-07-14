# Repository AI Agent Instructions

## Python environment

- NEVER create a Python virtual environment for this repository.
- Do not run `python -m venv`, `virtualenv`, `uv venv`, `conda create`, or any equivalent environment-creation command.
- Do not create `.venv`, `venv`, `.venv-*`, or other environment directories.
- Do not invoke VS Code's "Create Environment" workflow.
- Use the existing Homebrew Python 3.11 installation directly.
- When an explicit interpreter is required, use `/opt/homebrew/opt/python@3.11/bin/python3.11`.
- Install missing project dependencies into the existing Python environment when necessary.
- If a package version conflicts, diagnose and resolve the package conflict directly. Do not create an isolated environment as a workaround.
- Do not change the selected Python interpreter or Jupyter kernel unless execution is genuinely impossible with the existing Homebrew Python 3.11 interpreter.

## Repository safety

- Do not commit or push unless explicitly instructed.
- Do not add raw challenge data to Git.
- Do not modify organizer-provided starter files unless explicitly instructed.
- Prefer creating new exploratory files rather than editing organizer notebooks or source files during analysis.
- Treat Track 21 as held out for method and threshold tuning unless explicitly instructed otherwise.

## Scientific workflow

- Do not silently interpolate or fill missing profilometry values.
- Do not invent geometric targets when extraction fails.
- Preserve explicit validity and ambiguity states.
- Distinguish measurement or label confidence from predictive model uncertainty.
- Avoid track-specific thresholds or hacks unless explicitly requested and scientifically justified.
- Do not treat dense neighboring x positions as statistically independent samples.
- Prefer auditable diagnostics and quantitative comparisons before freezing preprocessing decisions.