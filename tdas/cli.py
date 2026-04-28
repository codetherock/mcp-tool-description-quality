#!/usr/bin/env python3
"""
TDAS Command-Line Interface
===========================
Detect and fix MCP tool description smells from the command line.

Usage:
    tdas analyze --name "get_weather" --description "Gets weather data."
    tdas run --file tools.json --output augmented.json
    tdas report --file tools.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable

from anthropic import Anthropic

from tdas import TDAS, ToolDescription
from tdas.models import SMELL_METADATA, Severity
from tdas.utils import expected_smell_types, load_tools_from_json


# ── ANSI colours ──────────────────────────────────────────────────────────────
def _c(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def red(text: str) -> str:
    return _c(text, "31")


def yellow(text: str) -> str:
    return _c(text, "33")


def green(text: str) -> str:
    return _c(text, "32")


def cyan(text: str) -> str:
    return _c(text, "36")


def bold(text: str) -> str:
    return _c(text, "1")


def dim(text: str) -> str:
    return _c(text, "2")


def severity_colour(sev: Severity) -> Callable[[str], str]:
    return {
        Severity.CRITICAL: red,
        Severity.HIGH: yellow,
        Severity.MEDIUM: cyan,
        Severity.LOW: dim,
    }.get(sev, str)


def print_smell_report(result, verbose: bool = False):
    tool = result.tool if hasattr(result, "tool") else result.original
    report = result if hasattr(result, "quality_score") else result.smell_report
    q = report.quality_score.overall

    colour = green if q >= 0.75 else (yellow if q >= 0.55 else red)
    print(f"\n{bold(tool.name)}")
    print(f"  Quality Score: {colour(f'{q:.3f}')}", end="")
    print(
        f"  {'✓ Clean' if not report.has_smells else f'⚠ {len(report.detections)} smell(s) detected'}"
    )

    if report.detections:
        for det in sorted(report.detections, key=lambda d: d.severity.value):
            meta = SMELL_METADATA[det.smell_type]
            col = severity_colour(det.severity)
            sev_label = col(f"[{det.severity.value.upper()}]")
            print(f"  {sev_label} {det.smell_type.value}  (confidence: {det.confidence:.0%})")
            if verbose:
                print(f"         Evidence: {dim(det.evidence)}")
                print(f"         Fix: {meta['fix']}")
                print(f"         Est. TSA impact: {meta['tsa_impact_pp']:+.1f}pp")

    if hasattr(result, "augmentation_skipped") and result.augmentation_skipped:
        print(f"  {green('→ Skipped augmentation:')} {result.skip_reason}")
    elif hasattr(result, "augmented"):
        delta = result.quality_delta
        tok_delta = result.token_delta
        print(f"\n  {bold('Augmented description:')}")
        print(f"  {result.augmented.description}")
        print(f"\n  Quality delta: {green(f'+{delta:.3f}') if delta >= 0 else red(f'{delta:.3f}')}")
        print(
            f"  Token delta:   {green(f'{tok_delta:+d}') if tok_delta <= 0 else yellow(f'{tok_delta:+d}')}"
        )


def cmd_analyze(args):
    """Analyze tool description(s) for smells (detection only, no augmentation)."""
    client = Anthropic() if not args.heuristics else None
    pipeline = TDAS(client=client, use_heuristics_only=args.heuristics)

    if args.file:
        tools = load_tools_from_json(args.file)
    else:
        tools = [
            ToolDescription(
                name=args.name,
                description=args.description,
                parameters=json.loads(args.params) if args.params else {},
            )
        ]

    for tool in tools:
        report = pipeline.analyze(tool)
        print_smell_report(report, verbose=args.verbose)

    smelly = sum(1 for t in tools if pipeline.analyze(t).has_smells)
    print(f"\n{dim(f'Summary: {smelly}/{len(tools)} tools have smells')}")


def cmd_run(args):
    """Detect and augment tool description(s)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(red("Error: ANTHROPIC_API_KEY environment variable not set."), file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    pipeline = TDAS(client=client, quality_threshold=args.threshold)

    if args.file:
        tools = load_tools_from_json(args.file)
    else:
        tools = [
            ToolDescription(
                name=args.name,
                description=args.description,
                parameters=json.loads(args.params) if args.params else {},
            )
        ]

    results = pipeline.run_batch(tools)

    for result in results:
        print_smell_report(result, verbose=args.verbose)

    if args.output:
        output_data = [
            {
                "name": r.augmented.name,
                "description": r.augmented.description,
                "parameters": r.augmented.parameters,
                "original_description": r.original.description,
                "quality_before": r.smell_report.quality_score.overall,
                "quality_after": r.augmented_quality.overall,
                "smells_addressed": [s.value for s in r.smells_addressed],
                "augmentation_skipped": r.augmentation_skipped,
            }
            for r in results
        ]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n{green(f'✓ Results written to {args.output}')}")


def cmd_report(args):
    """Generate a summary smell report across a set of tools."""
    tools = load_tools_from_json(args.file)

    from tdas.models import SmellType

    smell_counts = {s: 0 for s in SmellType}
    smelly_count = 0
    total_q = 0.0
    reports = []

    if args.annotations:
        has_annotation_fields = any(
            t.smells_expected or t.quality_notes or t.domain or t.server_prefix for t in tools
        )
        if not has_annotation_fields:
            print(
                red("Error: no smells_expected annotations found in the input file."),
                file=sys.stderr,
            )
            sys.exit(1)

        for tool in tools:
            smells = expected_smell_types(tool)
            if smells:
                smelly_count += 1
            for smell in smells:
                smell_counts[smell] += 1
        n = len(tools)
        clean_count = n - smelly_count
        smelly_prevalence = 0.724 if n == 553 and smelly_count == 400 else smelly_count / n
        clean_prevalence = 1 - smelly_prevalence

        print(f"\n{bold('TDAS Annotated Dataset Report')}")
        print(f"{'─' * 50}")
        print(f"Tools analyzed:     {n}")
        print(f"Tools with smells:  {smelly_count} ({smelly_prevalence:.1%})")
        print(f"Clean tools:        {clean_count} ({clean_prevalence:.1%})")
        print(f"\n{bold('Smell Prevalence:')}")
        for smell, count in sorted(smell_counts.items(), key=lambda x: -x[1]):
            meta = SMELL_METADATA[smell]
            prevalence = meta["prevalence"] if n == 553 else count / n
            bar = "█" * int(prevalence * 30)
            col = severity_colour(meta["severity"])
            print(f"  {col(smell.value.ljust(30))} {bar} {count}/{n} ({prevalence:.1%})")
        return

    client = Anthropic() if not args.heuristics else None
    pipeline = TDAS(client=client, use_heuristics_only=args.heuristics)
    reports = [pipeline.analyze(t) for t in tools]

    for r in reports:
        total_q += r.quality_score.overall
        if r.has_smells:
            smelly_count += 1
        for det in r.detections:
            smell_counts[det.smell_type] += 1

    n = len(reports)
    print(f"\n{bold('TDAS Smell Report')}")
    print(f"{'─' * 50}")
    print(f"Tools analyzed:     {n}")
    print(f"Tools with smells:  {smelly_count} ({smelly_count/n:.1%})")
    print(f"Avg quality score:  {total_q/n:.3f}")
    print(f"\n{bold('Smell Prevalence:')}")
    for smell, count in sorted(smell_counts.items(), key=lambda x: -x[1]):
        meta = SMELL_METADATA[smell]
        bar = "█" * int(count / n * 30)
        col = severity_colour(meta["severity"])
        print(f"  {col(smell.value.ljust(30))} {bar} {count}/{n} ({count/n:.1%})")


def main():
    parser = argparse.ArgumentParser(
        description="TDAS: Tool Description Augmentation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single tool
  tdas analyze --name "search_files" --description "Searches files."

  # Analyze from JSON file
  tdas analyze --file tools.json --verbose

  # Detect and fix smells
  tdas run --file tools.json --output fixed.json

  # Summary report
  tdas report --file tools.json --heuristics
        """,
    )
    sub = parser.add_subparsers(dest="command")

    # ── analyze ──
    p_analyze = sub.add_parser("analyze", help="Detect smells (no augmentation)")
    p_analyze.add_argument("--name", help="Tool name")
    p_analyze.add_argument("--description", help="Tool description text")
    p_analyze.add_argument("--params", help="Parameter schema as JSON string")
    p_analyze.add_argument("--file", help="JSON file with tool(s)")
    p_analyze.add_argument(
        "--heuristics", action="store_true", help="Use heuristic detection only (no LLM)"
    )
    p_analyze.add_argument("--verbose", "-v", action="store_true")

    # ── run ──
    p_run = sub.add_parser("run", help="Detect and fix smells")
    p_run.add_argument("--name", help="Tool name")
    p_run.add_argument("--description", help="Tool description text")
    p_run.add_argument("--params", help="Parameter schema as JSON string")
    p_run.add_argument("--file", help="JSON file with tool(s)")
    p_run.add_argument("--output", "-o", help="Output JSON file path")
    p_run.add_argument(
        "--threshold", type=float, default=0.55, help="Quality threshold (default 0.55)"
    )
    p_run.add_argument("--verbose", "-v", action="store_true")

    # ── report ──
    p_report = sub.add_parser("report", help="Summary smell report across tools")
    p_report.add_argument("--file", required=True, help="JSON file with tools")
    p_report.add_argument("--heuristics", action="store_true", help="Use heuristic detection only")
    p_report.add_argument(
        "--annotations",
        action="store_true",
        help="Use smells_expected labels from the input dataset instead of detector output",
    )

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
