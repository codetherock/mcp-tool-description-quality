"""
TDAS Utilities
==============
Helpers for loading MCP servers, formatting reports, and logging.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from tdas.models import (
    SHORT_SMELL_CODES,
    SMELL_METADATA,
    AugmentationResult,
    SmellReport,
    SmellType,
    ToolDescription,
)


def first_text_block(response: Any) -> str:
    """Return the first text block from an Anthropic response."""
    for block in response.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            return text
    raise ValueError("Anthropic response did not contain a text block.")


def expected_smell_types(tool: ToolDescription) -> list[SmellType]:
    """Return annotation smell labels, if present on a loaded dataset item."""
    smells = []
    for smell in tool.smells_expected:
        if isinstance(smell, SmellType):
            smells.append(smell)
        elif smell in SHORT_SMELL_CODES:
            smells.append(SHORT_SMELL_CODES[smell])
        else:
            try:
                smells.append(SmellType(smell))
            except ValueError:
                continue
    return smells


def load_tools_from_json(path: str) -> list[ToolDescription]:
    """
    Load tool descriptions from a JSON file.

    Accepts either a single tool dict or a list of tool dicts.
    Also accepts MCP server manifest format.

    Expected format:
        [
          {
            "name": "tool_name",
            "description": "...",
            "parameters": {"param1": {"type": "string"}, ...},
            "server_name": "optional",
            "server_url": "optional"
          }
        ]
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Might be an MCP server manifest
        if "tools" in data:
            data = data["tools"]
        else:
            data = [data]

    tools = []
    for item in data:
        tools.append(
            ToolDescription(
                name=item.get("name", "unknown"),
                description=item.get("description", ""),
                parameters=item.get(
                    "parameters", item.get("inputSchema", {}).get("properties", {})
                ),
                server_name=item.get("server_name"),
                server_url=item.get("server_url"),
                server_prefix=item.get("server_prefix"),
                domain=item.get("domain"),
                smells_expected=item.get("smells_expected", []),
                quality_notes=item.get("quality_notes"),
            )
        )
    return tools


def save_results_to_json(results: list[AugmentationResult], path: str):
    """Save augmentation results to JSON."""
    output = []
    for r in results:
        output.append(
            {
                "name": r.augmented.name,
                "original_description": r.original.description,
                "augmented_description": r.augmented.description,
                "parameters": r.augmented.parameters,
                "server_name": r.augmented.server_name,
                "server_prefix": r.augmented.server_prefix,
                "domain": r.augmented.domain,
                "smells_expected": r.augmented.smells_expected,
                "quality_before": r.smell_report.quality_score.overall,
                "quality_after": r.augmented_quality.overall,
                "quality_delta": r.quality_delta,
                "smells_detected": [d.smell_type.value for d in r.smell_report.detections],
                "smells_addressed": [s.value for s in r.smells_addressed],
                "token_delta": r.token_delta,
                "augmentation_skipped": r.augmentation_skipped,
                "skip_reason": r.skip_reason,
                "components": [
                    {
                        "component": c.component,
                        "label": c.label,
                        "score": c.score,
                        "rationale": c.rationale,
                    }
                    for c in r.smell_report.quality_score.components
                ],
            }
        )
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def format_report_text(report: SmellReport) -> str:
    """Format a SmellReport as a readable text summary."""
    lines = []
    q = report.quality_score.overall
    lines.append(f"Tool: {report.tool.name}")
    lines.append(
        f"Quality Score: {q:.3f} {'(AUGMENTATION NEEDED)' if report.quality_score.augmentation_needed else '(OK)'}"
    )
    lines.append(f"Token count: ~{report.tool.token_count()}")

    if report.detections:
        lines.append(f"\nDetected Smells ({len(report.detections)}):")
        for det in report.detections:
            meta = SMELL_METADATA[det.smell_type]
            lines.append(f"  [{det.severity.value.upper()}] {det.smell_type.value}")
            lines.append(f"    Evidence: {det.evidence}")
            lines.append(f"    Fix: {meta['fix']}")
            lines.append(f"    Est. TSA impact: {meta['tsa_impact_pp']:+.1f}pp")
    else:
        lines.append("\nNo smells detected. ✓")

    if report.quality_score.components:
        lines.append("\nComponent Scores (0-3):")
        for c in report.quality_score.components:
            bar = "■" * int(c.score) + "□" * (3 - int(c.score))
            lines.append(f"  {c.component} ({c.label:<28}) [{bar}] {c.score:.1f}/3.0")

    return "\n".join(lines)


def setup_logging(level: str = "INFO"):
    """Configure logging for TDAS."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
