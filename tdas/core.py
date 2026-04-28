"""
TDAS Core Pipeline
==================
The main entry point. Wires together detection and augmentation.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

from anthropic import Anthropic

from tdas.augmenters import Augmenter
from tdas.detectors import SmellDetector
from tdas.models import AugmentationResult, SmellReport, ToolDescription

logger = logging.getLogger(__name__)


class TDAS:
    """
    Tool Description Augmentation System.

    End-to-end pipeline for detecting and fixing MCP tool description smells.

    Usage:
        from anthropic import Anthropic
        from tdas import TDAS, ToolDescription

        client = Anthropic()
        pipeline = TDAS(client=client)

        tool = ToolDescription(
            name="get_weather",
            description="Gets weather.",
            parameters={"city": {"type": "string"}, "units": {"type": "string"}},
        )

        result = pipeline.run(tool)
        print(result.augmented.description)

    Args:
        client: Anthropic client. If None, uses heuristic-only detection (no augmentation).
        model: Model for detection and augmentation.
        quality_threshold: Q score below which augmentation is triggered. Default 0.55.
        use_heuristics_only: Use rule-based detection only (no LLM calls).
    """

    def __init__(
        self,
        client: Optional[Anthropic] = None,
        model: str = "claude-sonnet-4-6",
        quality_threshold: float = 0.55,
        use_heuristics_only: bool = False,
    ):
        self.client = client
        self.model = model
        self.quality_threshold = quality_threshold

        _heuristics_only = use_heuristics_only or client is None

        self.detector = SmellDetector(
            client=client,
            model=model,
            use_heuristics_only=_heuristics_only,
        )

        self.augmenter = (
            Augmenter(
                client=client,
                model=model,
                quality_threshold=quality_threshold,
                detector=self.detector,
            )
            if client and not _heuristics_only
            else None
        )

    def analyze(self, tool: Union[ToolDescription, dict]) -> SmellReport:
        """
        Analyze a tool description for smells without augmenting.

        Args:
            tool: ToolDescription or dict with keys: name, description, parameters (optional).

        Returns:
            SmellReport with quality score and all detected smells.
        """
        if isinstance(tool, dict):
            tool = ToolDescription(**tool)
        return self.detector.detect(tool)

    def run(self, tool: Union[ToolDescription, dict]) -> AugmentationResult:
        """
        Run full TDAS pipeline: detect smells and augment if needed.

        Args:
            tool: ToolDescription or dict with keys: name, description, parameters (optional).

        Returns:
            AugmentationResult with original, augmented description, and quality scores.

        Raises:
            RuntimeError: If augmentation is needed but no client was provided.
        """
        if isinstance(tool, dict):
            tool = ToolDescription(**tool)

        smell_report = self.detector.detect(tool)

        if (
            not smell_report.has_smells
            or smell_report.quality_score.overall >= self.quality_threshold
        ):
            # Clean description — return as-is
            return AugmentationResult(
                original=tool,
                augmented=tool,
                smell_report=smell_report,
                augmented_quality=smell_report.quality_score,
                smells_addressed=[],
                token_delta=0,
                augmentation_skipped=True,
                skip_reason="No smells detected or quality above threshold.",
            )

        if self.augmenter is None:
            raise RuntimeError(
                "Augmentation needed but no Anthropic client was provided. "
                "Initialize TDAS with client=Anthropic() to enable augmentation, "
                "or use analyze() for detection only."
            )

        return self.augmenter.augment(smell_report)

    def run_batch(
        self,
        tools: list[Union[ToolDescription, dict]],
        skip_clean: bool = True,
    ) -> list[AugmentationResult]:
        """
        Run TDAS on a batch of tools.

        Args:
            tools: List of ToolDescription objects or dicts.
            skip_clean: If True, skip augmentation for clean descriptions.

        Returns:
            List of AugmentationResult objects in the same order as input.
        """
        results = []
        for i, tool in enumerate(tools):
            if isinstance(tool, dict):
                tool = ToolDescription(**tool)
            logger.info(f"Processing tool {i+1}/{len(tools)}: {tool.name}")
            try:
                result = self.run(tool)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process tool '{tool.name}': {e}")
                raise
        return results
