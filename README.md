# TDAS — Tool Description Augmentation System

TDAS detects and improves weak natural-language descriptions for Model Context Protocol
(MCP) tools. MCP agents choose and call tools from text descriptions, so vague,
incomplete, or misleading descriptions can cause wrong tool selection, bad arguments,
extra retries, and higher token cost.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Problem TDAS Solves

LLM agents often see a registry of tools and must decide which one to call from
short natural-language descriptions. When those descriptions are vague, omit
constraints, fail to explain parameters, or lack examples, the agent can:

- choose the wrong tool,
- pass invalid arguments,
- retry failed calls,
- hallucinate capabilities a tool does not support,
- spend extra tokens on avoidable correction loops.

TDAS treats tool descriptions as quality-critical interface text. It gives teams
a repeatable way to lint descriptions, report common weaknesses, and optionally
rewrite poor descriptions before agents use them.

## Why It Is Useful

- **For MCP server maintainers:** catch weak tool descriptions before publishing a server.
- **For AI-agent builders:** improve tool-selection reliability without changing tool code.
- **For platform teams:** run TDAS as a CI quality gate over tool registries.
- **For evaluation:** compare raw descriptions against augmented descriptions with the included benchmark format.

TDAS supports two workflows:

| Workflow | Command | Use this when |
|---|---|---|
| Included benchmark report | `tdas report --annotations` | You want the prevalence table from the included labeled dataset |
| Live checking | `tdas analyze`, `tdas report --heuristics`, `tdas run` | You want to inspect or improve new MCP tool descriptions |

It checks six description smells:

| ID | Smell | Meaning | Included dataset prevalence |
|---|---|---|---:|
| S1 | Vague Intent | Purpose or domain is too unclear | 61.3% |
| S2 | Missing Constraints | Missing auth, rate limits, preconditions, or failures | 54.7% |
| S3 | Ambiguous Parameters | Parameter semantics, types, or formats are underspecified | 48.2% |
| S4 | Over-verbose | Description is too long or padded with irrelevant prose | 39.6% |
| S5 | Incomplete Examples | Missing concrete input/output examples | 72.4% |
| S6 | Misleading Scope | Description implies capabilities the tool does not have | 28.9% |

Included benchmark summary:

| Metric | Raw descriptions | TDAS | Change |
|---|---:|---:|---:|
| Task Success Rate | 0.51 | 0.74 | +45.1% |
| Tool Selection Accuracy | 0.58 | 0.81 | +39.7% |
| Token Cost, normalized | 1.00 | 0.61 | -39% |

## Repository Contents

```text
tdas/
├── tdas/                    # Python package
│   ├── cli.py               # Command-line interface
│   ├── core.py              # TDAS pipeline
│   ├── models.py            # Data models and smell metadata
│   ├── detectors/           # Heuristic and LLM-based smell detection
│   ├── augmenters/          # Smell-conditioned rewriting
│   ├── evaluation/          # Agent evaluation helpers
│   └── utils/               # JSON loading and report helpers
├── examples/
│   ├── tools.json           # Small demo dataset
│   ├── tools_553.json       # Included annotated dataset
│   └── demo.py              # Python API demo
├── tests/                   # Test suite
├── skills/tdas/SKILL.md     # AI-agent operating guide for this repo
├── pyproject.toml
└── README.md
```

## Step-by-Step Setup

### 1. Clone the repository

```bash
git clone https://github.com/codetherock/mcp-tool-description-quality.git
cd mcp-tool-description-quality
```

If you already have the folder locally, just `cd` into it.

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Do not commit `.venv/`; it is intentionally ignored by `.gitignore`.

### 3. Install TDAS for development

```bash
pip install -e ".[dev]"
```

This installs TDAS plus test and quality tools: `pytest`, `black`, `ruff`, and `mypy`.

### 4. Verify the CLI is available

```bash
tdas --help
```

You should see:

```text
usage: tdas [-h] {analyze,run,report} ...
TDAS: Tool Description Augmentation System
```

## Run the Included Benchmark Report

Use annotation mode to report the included 553-tool labeled dataset:

```bash
tdas report --file examples/tools_553.json --annotations
```

Expected output:

```text
TDAS Annotated Dataset Report
──────────────────────────────────────────────────
Tools analyzed:     553
Tools with smells:  400 (72.4%)
Clean tools:        153 (27.6%)

Smell Prevalence:
  S5_incomplete_examples         █████████████████████ 400/553 (72.4%)
  S1_vague_intent                ██████████████████ 339/553 (61.3%)
  S2_missing_constraints         ████████████████ 303/553 (54.7%)
  S3_ambiguous_parameters        ██████████████ 267/553 (48.2%)
  S4_over_verbose                ███████████ 219/553 (39.6%)
  S6_misleading_scope            ████████ 160/553 (28.9%)
```

`--annotations` reads the `smells_expected` labels stored in
`examples/tools_553.json`. Use this mode when you want the benchmark label
summary instead of recomputing smells with heuristics.

## Run Live Detection Without an API Key

Use heuristic mode for offline checking:

```bash
tdas analyze \
  --name "get_stock_price" \
  --description "Gets price." \
  --heuristics \
  --verbose
```

You can also run heuristics over a JSON file:

```bash
tdas report --file examples/tools.json --heuristics
```

Use `--heuristics` when the input file does not have annotation labels.

## Run the Demo Script

```bash
python examples/demo.py
```

The demo always runs offline examples first. LLM-based augmentation examples are
skipped unless `ANTHROPIC_API_KEY` is set.

## Run LLM-Based Augmentation

Augmentation rewrites smelly descriptions using smell-specific prompts. It
requires an Anthropic API key.

macOS/Linux:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Windows PowerShell:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Then run:

```bash
tdas run \
  --file examples/tools.json \
  --output augmented_tools.json \
  --verbose
```

For cost control, start with `examples/tools.json` before running the full
553-tool dataset.

## Run Tests and Quality Checks

Before committing:

```bash
ruff check tdas/
black --check tdas/
mypy tdas
pytest tests -v
```

Expected test result:

```text
20 passed
```

## Input JSON Format

TDAS accepts a list of tools:

```json
[
  {
    "name": "create_github_issue",
    "description": "Creates a GitHub issue.",
    "parameters": {
      "owner": {"type": "string"},
      "repo": {"type": "string"},
      "title": {"type": "string"},
      "body": {"type": "string"}
    },
    "server_name": "DevOpsMCP"
  }
]
```

Annotated datasets can also include optional metadata:

```json
{
  "domain": "devops",
  "server_prefix": "ToolBench",
  "smells_expected": ["S1", "S5"],
  "quality_notes": "Smells: S1, S5"
}
```

## Python API

Offline analysis:

```python
from tdas import TDAS, ToolDescription

pipeline = TDAS(use_heuristics_only=True)

tool = ToolDescription(
    name="get_stock_price",
    description="Gets price.",
    parameters={"ticker": {"type": "string"}},
)

report = pipeline.analyze(tool)
print(report.quality_score.overall)
print([d.smell_type.value for d in report.detections])
```

LLM augmentation:

```python
from anthropic import Anthropic
from tdas import TDAS, ToolDescription

pipeline = TDAS(client=Anthropic())

result = pipeline.run(ToolDescription(
    name="get_stock_price",
    description="Gets price.",
    parameters={"ticker": {"type": "string"}},
))

print(result.augmented.description)
print(result.quality_delta)
```

## How It Works

1. A tool description is loaded from JSON or Python objects.
2. The detector scores six quality components on a 0-3 scale.
3. Low component scores map to S1-S6 smell labels.
4. Descriptions below the quality threshold can be rewritten.
5. The augmenter uses smell-specific prompts instead of generic paraphrasing.
6. Reports summarize quality scores, smell prevalence, and augmentation results.

Quality score:

```text
Q(d) = (0.22*C1 + 0.18*C2 + 0.17*C3 + 0.10*C4 + 0.21*C5 + 0.12*C6) / 3
```

Descriptions with `Q < 0.55` are candidates for augmentation.

## Notes for GitHub

Commit source, tests, docs, and examples:

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

Do not commit generated or local files:

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

## Related Work and Novelty

TDAS builds on existing ideas from code smells, API documentation quality,
prompt quality, and LLM tool-use benchmarks. The contribution here is applying
a smell taxonomy to MCP tool descriptions and pairing smell detection with
smell-conditioned rewriting for agent-facing tool metadata.

## License

MIT License. See [LICENSE](LICENSE).
