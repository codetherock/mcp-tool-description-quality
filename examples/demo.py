#!/usr/bin/env python3
"""
TDAS Usage Examples
===================
Demonstrates the main TDAS API patterns.
"""

import os
from anthropic import Anthropic
from tdas import TDAS, ToolDescription
from tdas.utils import format_report_text, load_tools_from_json, save_results_to_json


def example_1_analyze_single_tool():
    """Example 1: Analyze a single tool for smells (no augmentation)."""
    print("\n" + "="*60)
    print("Example 1: Analyze a single tool (heuristic mode)")
    print("="*60)

    pipeline = TDAS(use_heuristics_only=True)  # No API key needed

    tool = ToolDescription(
        name="search_products",
        description="Searches products.",
        parameters={"query": {"type": "string"}, "category": {"type": "string"}},
    )

    report = pipeline.analyze(tool)
    print(format_report_text(report))


def example_2_full_augmentation():
    """Example 2: Detect and fix smells using the Anthropic API."""
    print("\n" + "="*60)
    print("Example 2: Full detection + augmentation pipeline")
    print("="*60)

    client = Anthropic()
    pipeline = TDAS(client=client)

    tool = ToolDescription(
        name="get_stock_price",
        description="Gets price.",
        parameters={"ticker": {"type": "string"}},
    )

    result = pipeline.run(tool)

    print(f"Original:  {result.original.description}")
    print(f"Augmented: {result.augmented.description}")
    print(f"Quality:   {result.smell_report.quality_score.overall:.3f} → {result.augmented_quality.overall:.3f}")
    print(f"Smells fixed: {[s.value for s in result.smells_addressed]}")


def example_3_batch_from_file():
    """Example 3: Process a JSON file of tools."""
    print("\n" + "="*60)
    print("Example 3: Batch processing from file")
    print("="*60)

    client = Anthropic()
    pipeline = TDAS(client=client)

    tools = load_tools_from_json("examples/tools.json")
    print(f"Loaded {len(tools)} tools")

    results = pipeline.run_batch(tools)

    for r in results:
        status = "✓ SKIPPED" if r.augmentation_skipped else f"↑ {r.quality_delta:+.3f}"
        print(f"  {r.original.name}: {status}")

    save_results_to_json(results, "examples/augmented_tools.json")
    print("\nResults saved to examples/augmented_tools.json")


def example_4_analyze_only_heuristics():
    """Example 4: Fast heuristic-only smell report — no API key needed."""
    print("\n" + "="*60)
    print("Example 4: Heuristic report (offline mode)")
    print("="*60)

    pipeline = TDAS(use_heuristics_only=True)
    tools = load_tools_from_json("examples/tools.json")

    smelly = 0
    for tool in tools:
        report = pipeline.analyze(tool)
        if report.has_smells:
            smelly += 1
            print(f"  ⚠ {tool.name}: {len(report.detections)} smell(s), Q={report.quality_score.overall:.3f}")
        else:
            print(f"  ✓ {tool.name}: Q={report.quality_score.overall:.3f}")

    print(f"\n{smelly}/{len(tools)} tools need attention")


if __name__ == "__main__":
    print("TDAS Usage Examples")

    # Example 1 always runs (no API key needed)
    example_1_analyze_single_tool()
    example_4_analyze_only_heuristics()

    # Examples 2 & 3 require ANTHROPIC_API_KEY
    if os.environ.get("ANTHROPIC_API_KEY"):
        example_2_full_augmentation()
        example_3_batch_from_file()
    else:
        print("\n[Skipping Examples 2 & 3: set ANTHROPIC_API_KEY to run augmentation]")
