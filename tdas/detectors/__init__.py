"""
TDAS Smell Detector
===================
Detects the six MCP tool description smells (S1-S6) using a combination of
rule-based heuristics and LLM-based component scoring.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from anthropic import Anthropic

from tdas.models import (
    SMELL_METADATA,
    ComponentScore,
    QualityScore,
    SmellDetection,
    SmellReport,
    SmellType,
    ToolDescription,
)
from tdas.utils import first_text_block

logger = logging.getLogger(__name__)


# Component thresholds — scores below these trigger the associated smell
COMPONENT_THRESHOLDS = {
    "C1": 1.5,  # Intent Clarity       → S1
    "C2": 1.5,  # Constraint Complete  → S2
    "C3": 1.5,  # Parameter Precision  → S3
    "C4": 1.5,  # Length Appropriate   → S4
    "C5": 1.0,  # Example Coverage     → S5 (stricter)
    "C6": 1.5,  # Scope Accuracy       → S6
}

COMPONENT_TO_SMELL = {
    "C1": SmellType.S1_VAGUE_INTENT,
    "C2": SmellType.S2_MISSING_CONSTRAINTS,
    "C3": SmellType.S3_AMBIGUOUS_PARAMETERS,
    "C4": SmellType.S4_OVER_VERBOSE,
    "C5": SmellType.S5_INCOMPLETE_EXAMPLES,
    "C6": SmellType.S6_MISLEADING_SCOPE,
}

COMPONENT_LABELS = {
    "C1": "Intent Clarity",
    "C2": "Constraint Completeness",
    "C3": "Parameter Precision",
    "C4": "Length Appropriateness",
    "C5": "Example Coverage",
    "C6": "Scope Accuracy",
}

SCORING_SYSTEM_PROMPT = """You are an expert evaluator of MCP (Model Context Protocol) tool descriptions.
Your task is to score a tool description on six quality components, each on a 0-3 scale.

Scoring rubric:
- 0 = Absent: The component is completely missing
- 1 = Partial: The component is present but severely lacking
- 2 = Adequate: The component meets minimum quality standards
- 3 = Exemplary: The component is thorough and clear

The six components are:
C1 - Intent Clarity: Does the description unambiguously state the tool's primary function and target domain?
C2 - Constraint Completeness: Are pre-conditions, rate limits, auth requirements, and failure modes documented?
C3 - Parameter Precision: Are all parameters named, typed, and semantically explained with format examples?
C4 - Length Appropriateness: Is description length proportional to tool complexity? (Ideal: 50-150 tokens. Penalize stubs <20 tokens and walls of text >200 tokens)
C5 - Example Coverage: Does the description include at least one concrete input/output example?
C6 - Scope Accuracy: Does the description accurately bound the tool's capabilities (not over-promise)?

Respond ONLY with a valid JSON object. No preamble, no markdown fences. Example:
{
  "C1": {"score": 2, "rationale": "Function is clear but domain scope is not specified."},
  "C2": {"score": 0, "rationale": "No constraints, rate limits, or auth requirements mentioned."},
  "C3": {"score": 1, "rationale": "Parameters listed but no type or format examples given."},
  "C4": {"score": 3, "rationale": "Description is concise and appropriately detailed."},
  "C5": {"score": 0, "rationale": "No input/output examples provided."},
  "C6": {"score": 2, "rationale": "Scope is mostly accurate but slightly overstated."}
}"""


class SmellDetector:
    """
    Detects MCP tool description smells using LLM-based component scoring.

    Args:
        client: Anthropic client. If None, uses heuristic-only detection.
        model: Model to use for scoring.
        use_heuristics_only: Skip LLM calls and use rule-based detection only.
    """

    def __init__(
        self,
        client: Optional[Anthropic] = None,
        model: str = "claude-sonnet-4-6",
        use_heuristics_only: bool = False,
    ):
        self.client = client
        self.model = model
        self.use_heuristics_only = use_heuristics_only or client is None

    def detect(self, tool: ToolDescription) -> SmellReport:
        """
        Run full smell detection on a tool description.

        Returns a SmellReport with component scores, quality score, and
        all detected smells.
        """
        if self.use_heuristics_only:
            components = self._heuristic_score(tool)
        else:
            components = self._llm_score(tool)

        quality = QualityScore.from_components(components)
        detections = self._build_detections(components)
        smell_set = [d.smell_type for d in detections]

        return SmellReport(
            tool=tool,
            quality_score=quality,
            detections=detections,
            smell_set=smell_set,
        )

    def _llm_score(self, tool: ToolDescription) -> list[ComponentScore]:
        """Use LLM to score the six quality components."""
        param_info = ""
        if tool.parameters:
            param_info = f"\n\nParameter schema:\n{json.dumps(tool.parameters, indent=2)}"

        user_prompt = f"""Tool name: {tool.name}

Tool description:
{tool.description}{param_info}

Score this tool description on all six quality components (C1-C6)."""

        try:
            assert self.client is not None
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=SCORING_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = first_text_block(response).strip()
            # Strip markdown fences if present
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            scores = json.loads(raw)
            return self._parse_llm_scores(scores)
        except Exception as e:
            logger.warning(
                f"LLM scoring failed for '{tool.name}': {e}. Falling back to heuristics."
            )
            return self._heuristic_score(tool)

    def _parse_llm_scores(self, scores: dict) -> list[ComponentScore]:
        components = []
        for key in ["C1", "C2", "C3", "C4", "C5", "C6"]:
            val = scores.get(key, {})
            score = float(val.get("score", 0))
            rationale = val.get("rationale", "")
            components.append(
                ComponentScore(
                    component=key,
                    label=COMPONENT_LABELS[key],
                    score=min(max(score, 0.0), 3.0),
                    rationale=rationale,
                )
            )
        return components

    def _heuristic_score(self, tool: ToolDescription) -> list[ComponentScore]:
        """
        Rule-based heuristic scoring as fallback when no LLM is available.
        Less accurate than LLM scoring but requires no API calls.
        """
        desc = tool.description.strip()
        words = desc.split()
        tokens = int(len(words) * 1.3)

        # C1: Intent Clarity
        vague_words = {
            "thing",
            "stuff",
            "something",
            "various",
            "etc",
            "data",
            "info",
            "handle",
            "process",
            "manage",
        }
        vague_count = sum(1 for w in words if w.lower() in vague_words)
        c1 = (
            3.0
            if (len(words) > 15 and vague_count == 0)
            else (2.0 if len(words) > 8 else (1.0 if len(words) > 3 else 0.0))
        )
        c1 = max(0.0, c1 - vague_count * 0.5)

        # C2: Constraint Completeness
        constraint_signals = [
            "require",
            "must",
            "only",
            "limit",
            "rate",
            "auth",
            "token",
            "key",
            "permission",
            "fail",
            "error",
            "exception",
            "throw",
            "raise",
            "invalid",
            "precondition",
        ]
        c2_hits = sum(1 for s in constraint_signals if s in desc.lower())
        c2 = min(3.0, c2_hits * 0.6)

        # C3: Parameter Precision
        has_params = bool(tool.parameters)
        param_mentions = (
            sum(1 for p in tool.parameters if p.lower() in desc.lower()) if has_params else 0
        )
        format_signals = [
            "format",
            "e.g.",
            "example",
            "yyyy",
            "iso",
            "json",
            "string",
            "integer",
            "boolean",
            "list",
            "array",
        ]
        format_hits = sum(1 for s in format_signals if s in desc.lower())
        c3 = (
            0.0
            if not has_params
            else min(3.0, (param_mentions / max(len(tool.parameters), 1)) * 2 + format_hits * 0.3)
        )

        # C4: Length Appropriateness
        if tokens < 10:
            c4 = 0.0
        elif tokens < 20:
            c4 = 1.0
        elif 30 <= tokens <= 120:
            c4 = 3.0
        elif tokens <= 160:
            c4 = 2.0
        elif tokens <= 200:
            c4 = 1.0
        else:
            c4 = max(0.0, 1.0 - (tokens - 200) / 100)

        # C5: Example Coverage
        example_signals = [
            "e.g.",
            "for example",
            "example:",
            "input:",
            "output:",
            "returns:",
            "→",
            "->",
            '"""',
            "```",
            "e.g,",
            "such as",
        ]
        c5 = 3.0 if any(s in desc.lower() for s in example_signals) else 0.0

        # C6: Scope Accuracy
        overscope_signals = [
            "anything",
            "everything",
            "any task",
            "all",
            "fully",
            "completely",
            "unlimited",
            "any kind",
            "general purpose",
            "universal",
        ]
        overscope_hits = sum(1 for s in overscope_signals if s in desc.lower())
        c6 = max(0.0, 3.0 - overscope_hits * 1.0)

        return [
            ComponentScore("C1", COMPONENT_LABELS["C1"], round(c1, 2), "Heuristic-based"),
            ComponentScore("C2", COMPONENT_LABELS["C2"], round(c2, 2), "Heuristic-based"),
            ComponentScore("C3", COMPONENT_LABELS["C3"], round(c3, 2), "Heuristic-based"),
            ComponentScore("C4", COMPONENT_LABELS["C4"], round(c4, 2), "Heuristic-based"),
            ComponentScore("C5", COMPONENT_LABELS["C5"], round(c5, 2), "Heuristic-based"),
            ComponentScore("C6", COMPONENT_LABELS["C6"], round(c6, 2), "Heuristic-based"),
        ]

    def _build_detections(self, components: list[ComponentScore]) -> list[SmellDetection]:
        detections = []
        for comp in components:
            threshold = COMPONENT_THRESHOLDS[comp.component]
            if comp.score < threshold:
                smell_type = COMPONENT_TO_SMELL[comp.component]
                meta = SMELL_METADATA[smell_type]
                confidence = 1.0 - (comp.score / threshold)
                detections.append(
                    SmellDetection(
                        smell_type=smell_type,
                        severity=meta["severity"],
                        confidence=round(confidence, 3),
                        evidence=comp.rationale,
                        component_score=comp.score,
                        threshold=threshold,
                    )
                )
        return detections
