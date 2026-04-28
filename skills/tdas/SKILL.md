---
name: tdas
description: Use this skill when working with the TDAS repository, running TDAS CLI commands, checking MCP tool description smells, using the included annotated benchmark dataset, augmenting tool descriptions, or preparing this repo for commit/release.
---

# TDAS Repository Skill

## Purpose

TDAS is a Python CLI/package for detecting and remediating MCP tool description smells.
Use this skill to operate the repo consistently and avoid confusing annotation-backed reports with live detector output.

TDAS solves the practical problem of weak tool descriptions causing LLM agents to select the wrong tool, pass invalid arguments, or waste tokens on failed calls and retries.

## First Checks

From the repo root:

```bash
pwd
find . -maxdepth 2 -type f | sort
```

If dependencies are missing, create a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Never commit `.venv/`, `dist/`, `build/`, `*.egg-info/`, caches, or coverage files.

## Annotated Benchmark Report

Use annotation mode:

```bash
tdas report --file examples/tools_553.json --annotations
```

Expected benchmark summary:

```text
Tools analyzed:     553
Tools with smells:  400 (72.4%)
Clean tools:        153 (27.6%)
S5: 400/553 (72.4%)
S1: 339/553 (61.3%)
S2: 303/553 (54.7%)
S3: 267/553 (48.2%)
S4: 219/553 (39.6%)
S6: 160/553 (28.9%)
```

Important: `--annotations` uses `smells_expected` labels in `examples/tools_553.json`.
It is the correct mode for reporting the included labeled dataset.

## Live Detection

Use heuristic mode for new or unlabeled tool descriptions:

```bash
tdas analyze --name "get_stock_price" --description "Gets price." --heuristics --verbose
tdas report --file examples/tools.json --heuristics
```

Heuristic output may differ from the annotated report because it recomputes smells instead of reading labels.

## Augmentation

LLM augmentation requires `ANTHROPIC_API_KEY`:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
tdas run --file examples/tools.json --output augmented_tools.json --verbose
```

Start with `examples/tools.json` before running the 553-tool dataset to avoid unnecessary API cost.

## Validation Before Commit

Run:

```bash
ruff check tdas/
black --check tdas/
mypy tdas
pytest tests -v
tdas report --file examples/tools_553.json --annotations
```

Expected tests: `20 passed`.

## Key Files

- `tdas/cli.py`: CLI commands and report modes.
- `tdas/models.py`: smell metadata, quality score weights, tool/result models.
- `tdas/detectors/__init__.py`: heuristic and LLM scoring.
- `tdas/augmenters/__init__.py`: smell-conditioned rewriting.
- `tdas/utils/__init__.py`: JSON loading, annotation helpers, report formatting.
- `examples/tools_553.json`: included annotated benchmark dataset.
- `README.md`: user-facing setup and run instructions.

## GitHub Hygiene

Commit:

```text
.github/
examples/
skills/
tdas/
tests/
.gitignore
LICENSE
README.md
pyproject.toml
```

Do not commit:

```text
.venv/
dist/
build/
*.egg-info/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
.DS_Store
```
