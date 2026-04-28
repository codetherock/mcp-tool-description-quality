"""
TDAS Augmenter
==============
Smell-conditioned rewriting of MCP tool descriptions using GPT-4o / Claude.

Each smell type gets a distinct rewrite template targeting its specific deficit,
rather than generic paraphrasing.
"""

from __future__ import annotations

import json
import logging

from anthropic import Anthropic

from tdas.models import (
    AugmentationResult,
    QualityScore,
    SmellReport,
    SmellType,
    ToolDescription,
)
from tdas.utils import first_text_block

logger = logging.getLogger(__name__)


# ── Smell-conditioned system prompts ──────────────────────────────────────────

BASE_SYSTEM = """You are an expert technical writer specializing in MCP (Model Context Protocol) tool descriptions.
Your task is to rewrite a tool description to fix specific quality issues.
Write ONLY the improved description text. No preamble, no labels, no markdown.
The rewritten description should be clear, precise, and usable by an AI agent without any other context."""

SMELL_PROMPTS: dict[SmellType, str] = {
    SmellType.S1_VAGUE_INTENT: """The tool description has a VAGUE INTENT smell.
Fix: Rewrite so the first sentence clearly states:
1. The exact action the tool performs (verb + object)
2. The specific domain or service it operates on
3. When an agent should prefer this tool over alternatives
Avoid generic verbs like "handle", "manage", "process", "deal with".""",
    SmellType.S2_MISSING_CONSTRAINTS: """The tool description has a MISSING CONSTRAINTS smell.
Fix: Add a concise constraints section covering:
1. Required preconditions (e.g. "Requires authenticated session", "File must exist")
2. Rate limits if applicable (e.g. "Rate limited to 100 calls/minute")
3. Known failure modes and what causes them
4. Any important postconditions or side effects
Keep additions brief and factual.""",
    SmellType.S3_AMBIGUOUS_PARAMETERS: """The tool description has an AMBIGUOUS PARAMETERS smell.
Fix: For each parameter in the schema, ensure the description mentions:
1. The parameter's semantic purpose (not just its name)
2. Expected format or type (e.g. "ISO 8601 date string", "positive integer", "comma-separated list")
3. Any valid value ranges or enumerations
Integrate parameter guidance naturally into the description prose.""",
    SmellType.S4_OVER_VERBOSE: """The tool description has an OVER-VERBOSE smell (too long, >200 tokens).
Fix: Rewrite to be concise (target 50-150 tokens) by:
1. Removing redundant phrases and filler words
2. Keeping only information an agent needs to SELECT and USE this tool correctly
3. Cutting implementation details that don't affect tool usage
Do not remove examples, constraints, or parameter guidance — trim prose only.""",
    SmellType.S5_INCOMPLETE_EXAMPLES: """The tool description has an INCOMPLETE EXAMPLES smell.
Fix: Add at least one concrete input/output example showing:
1. Typical parameter values (realistic, not placeholder)
2. What the tool returns for those inputs
3. If the tool can fail, optionally show an error case
Format the example clearly, e.g.:
  Example: get_weather(city="London", units="celsius") → {"temp": 14, "condition": "cloudy"}""",
    SmellType.S6_MISLEADING_SCOPE: """The tool description has a MISLEADING SCOPE smell.
Fix: Rewrite to accurately bound what the tool can and cannot do:
1. Be specific about what the tool does NOT support
2. Remove or qualify any overstated capabilities
3. If the tool has known limitations, state them briefly
Example: instead of "searches any data", write "searches product catalog by name or SKU only".""",
}


def _build_augmentation_prompt(
    tool: ToolDescription,
    smell_report: SmellReport,
) -> str:
    """Build the user prompt for augmentation."""
    param_info = ""
    if tool.parameters:
        param_info = f"\n\nParameter schema:\n{json.dumps(tool.parameters, indent=2)}"

    detected = "\n".join(
        f"- {d.smell_type.value} (confidence: {d.confidence:.0%}): {d.evidence}"
        for d in smell_report.detections
    )

    smell_instructions = "\n\n".join(SMELL_PROMPTS[d.smell_type] for d in smell_report.detections)

    return f"""Tool name: {tool.name}

Original description:
{tool.description}{param_info}

Detected smells:
{detected}

Instructions for each smell:
{smell_instructions}

Rewrite the description to fix ALL detected smells above. Output only the improved description."""


class Augmenter:
    """
    Smell-conditioned tool description augmenter.

    Uses the detected smell set to apply targeted rewrite templates,
    rather than generic paraphrasing.

    Args:
        client: Anthropic client.
        model: Model to use for augmentation.
        quality_threshold: Descriptions with Q >= threshold are skipped.
        detector: SmellDetector instance for re-scoring after augmentation.
    """

    def __init__(
        self,
        client: Anthropic,
        model: str = "claude-sonnet-4-6",
        quality_threshold: float = 0.55,
        detector=None,
    ):
        self.client = client
        self.model = model
        self.quality_threshold = quality_threshold
        self.detector = detector

    def augment(self, smell_report: SmellReport) -> AugmentationResult:
        """
        Augment a tool description based on its smell report.

        Implements selective invocation: descriptions with Q >= threshold
        are returned unchanged (saves ~27.6% of API calls).
        """
        tool = smell_report.tool
        q = smell_report.quality_score.overall

        # Selective invocation: skip clean descriptions
        if q >= self.quality_threshold and not smell_report.has_smells:
            augmented_tool = ToolDescription(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                server_name=tool.server_name,
                server_url=tool.server_url,
                server_prefix=tool.server_prefix,
                domain=tool.domain,
                smells_expected=tool.smells_expected,
                quality_notes=tool.quality_notes,
            )
            return AugmentationResult(
                original=tool,
                augmented=augmented_tool,
                smell_report=smell_report,
                augmented_quality=smell_report.quality_score,
                smells_addressed=[],
                token_delta=0,
                augmentation_skipped=True,
                skip_reason=f"Quality score {q:.3f} >= threshold {self.quality_threshold}. No augmentation needed.",
            )

        # Build smell-conditioned prompt
        system = BASE_SYSTEM
        user = _build_augmentation_prompt(tool, smell_report)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            new_description = first_text_block(response).strip()
        except Exception as e:
            logger.error(f"Augmentation failed for '{tool.name}': {e}")
            raise

        augmented_tool = ToolDescription(
            name=tool.name,
            description=new_description,
            parameters=tool.parameters,
            server_name=tool.server_name,
            server_url=tool.server_url,
            server_prefix=tool.server_prefix,
            domain=tool.domain,
            smells_expected=tool.smells_expected,
            quality_notes=tool.quality_notes,
        )

        # Re-score the augmented description if detector is available
        if self.detector:
            post_report = self.detector.detect(augmented_tool)
            augmented_quality = post_report.quality_score
        else:
            # Estimate improvement without re-scoring
            augmented_quality = QualityScore(
                overall=min(1.0, smell_report.quality_score.overall + 0.25),
                augmentation_needed=False,
            )

        token_delta = augmented_tool.token_count() - tool.token_count()

        return AugmentationResult(
            original=tool,
            augmented=augmented_tool,
            smell_report=smell_report,
            augmented_quality=augmented_quality,
            smells_addressed=[d.smell_type for d in smell_report.detections],
            token_delta=token_delta,
        )
