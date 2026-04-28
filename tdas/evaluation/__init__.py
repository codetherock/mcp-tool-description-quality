"""
TDAS Evaluation
===============
Tools for evaluating tool selection accuracy and augmentation quality.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from anthropic import Anthropic

from tdas.models import ToolDescription
from tdas.utils import first_text_block

logger = logging.getLogger(__name__)


@dataclass
class EvalTask:
    """A single evaluation task."""

    task_id: str
    query: str  # Natural language task description
    correct_tool: str  # Name of the correct tool to select
    correct_args: dict = field(default_factory=dict)  # Correct arguments
    domain: str = "general"


@dataclass
class EvalResult:
    """Result of evaluating one task."""

    task: EvalTask
    selected_tool: str
    selected_args: dict
    tool_selection_correct: bool
    argument_f1: float
    task_success: bool
    tokens_used: int = 0
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Aggregate results across all tasks."""

    results: list[EvalResult]
    condition: str  # "original" or "augmented"

    @property
    def task_success_rate(self) -> float:
        return sum(r.task_success for r in self.results) / len(self.results)

    @property
    def tool_selection_accuracy(self) -> float:
        return sum(r.tool_selection_correct for r in self.results) / len(self.results)

    @property
    def mean_argument_f1(self) -> float:
        return sum(r.argument_f1 for r in self.results) / len(self.results)

    @property
    def mean_tokens(self) -> float:
        return sum(r.tokens_used for r in self.results) / len(self.results)

    def summary(self) -> dict:
        return {
            "condition": self.condition,
            "n_tasks": len(self.results),
            "task_success_rate": round(self.task_success_rate, 4),
            "tool_selection_accuracy": round(self.tool_selection_accuracy, 4),
            "mean_argument_f1": round(self.mean_argument_f1, 4),
            "mean_tokens": round(self.mean_tokens, 1),
        }


TOOL_SELECTION_SYSTEM = """You are an AI agent with access to a set of tools.
Given a user task, select the most appropriate tool and provide its arguments.

Respond ONLY with a JSON object:
{
  "tool": "<tool_name>",
  "arguments": {<key>: <value>, ...},
  "reasoning": "<brief explanation>"
}"""


def _compute_argument_f1(predicted: dict, reference: dict) -> float:
    """Compute token-level F1 between predicted and reference argument sets."""
    if not reference and not predicted:
        return 1.0
    if not reference or not predicted:
        return 0.0

    pred_pairs = set(f"{k}={v}" for k, v in predicted.items())
    ref_pairs = set(f"{k}={v}" for k, v in reference.items())

    tp = len(pred_pairs & ref_pairs)
    precision = tp / len(pred_pairs) if pred_pairs else 0.0
    recall = tp / len(ref_pairs) if ref_pairs else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class Evaluator:
    """
    Evaluates agent performance with original vs augmented tool descriptions.

    Implements a task-based evaluation protocol across domains.
    """

    def __init__(self, client: Anthropic, model: str = "claude-sonnet-4-6"):
        self.client = client
        self.model = model

    def evaluate_task(
        self,
        task: EvalTask,
        tools: list[ToolDescription],
    ) -> EvalResult:
        """Evaluate a single task with a given set of tool descriptions."""
        tool_registry = self._format_registry(tools)

        user_prompt = f"""Available tools:
{tool_registry}

Task: {task.query}

Select the appropriate tool and provide arguments."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                system=TOOL_SELECTION_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = first_text_block(response).strip()
            import re

            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            parsed = json.loads(raw)

            selected_tool = parsed.get("tool", "")
            selected_args = parsed.get("arguments", {})
            tokens_used = response.usage.input_tokens + response.usage.output_tokens

        except Exception as e:
            logger.warning(f"Agent call failed for task {task.task_id}: {e}")
            return EvalResult(
                task=task,
                selected_tool="",
                selected_args={},
                tool_selection_correct=False,
                argument_f1=0.0,
                task_success=False,
                error=str(e),
            )

        tool_correct = selected_tool == task.correct_tool
        arg_f1 = _compute_argument_f1(selected_args, task.correct_args)
        # Task success = correct tool AND argument F1 >= 0.8
        task_success = tool_correct and arg_f1 >= 0.8

        return EvalResult(
            task=task,
            selected_tool=selected_tool,
            selected_args=selected_args,
            tool_selection_correct=tool_correct,
            argument_f1=arg_f1,
            task_success=task_success,
            tokens_used=tokens_used,
        )

    def compare(
        self,
        tasks: list[EvalTask],
        original_tools: list[ToolDescription],
        augmented_tools: list[ToolDescription],
    ) -> tuple[BenchmarkResult, BenchmarkResult]:
        """
        Compare agent performance with original vs augmented descriptions.

        Returns (original_results, augmented_results).
        """
        original_results = []
        augmented_results = []

        for task in tasks:
            orig = self.evaluate_task(task, original_tools)
            aug = self.evaluate_task(task, augmented_tools)
            original_results.append(orig)
            augmented_results.append(aug)

        return (
            BenchmarkResult(original_results, condition="original"),
            BenchmarkResult(augmented_results, condition="augmented"),
        )

    def _format_registry(self, tools: list[ToolDescription]) -> str:
        lines = []
        for tool in tools:
            lines.append(f"Tool: {tool.name}")
            lines.append(f"Description: {tool.description}")
            if tool.parameters:
                lines.append(f"Parameters: {json.dumps(tool.parameters)}")
            lines.append("")
        return "\n".join(lines)
