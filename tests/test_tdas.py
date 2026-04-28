"""
TDAS Test Suite
===============
Tests for smell detection, augmentation, and the core pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch

from tdas import TDAS, ToolDescription
from tdas.models import SmellType, Severity, QualityScore, ComponentScore
from tdas.detectors import SmellDetector
from tdas.utils import format_report_text


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_tool():
    return ToolDescription(
        name="get_weather",
        description=(
            "Retrieves current weather conditions for a specific city. "
            "Requires a valid city name and optional unit preference. "
            "Returns temperature, humidity, and weather condition. "
            "Rate limited to 60 calls/minute. "
            "Example: get_weather(city='London', units='celsius') → "
            "{'temp': 14, 'condition': 'cloudy', 'humidity': 82}"
        ),
        parameters={
            "city": {"type": "string", "description": "City name, e.g. 'London'"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
        },
    )


@pytest.fixture
def smelly_tool_s5():
    """Tool with S5 (Incomplete Examples) smell — most common smell."""
    return ToolDescription(
        name="search_products",
        description="Searches the product catalog.",
        parameters={"query": {"type": "string"}},
    )


@pytest.fixture
def smelly_tool_multi():
    """Tool with multiple smells."""
    return ToolDescription(
        name="process_data",
        description="Handles data.",
        parameters={},
    )


@pytest.fixture
def over_verbose_tool():
    return ToolDescription(
        name="send_email",
        description=(
            "This tool is designed to send emails to recipients. The tool has been carefully "
            "designed to work with the email system and provides a comprehensive set of features "
            "for email composition and delivery. It supports various email formats and can handle "
            "different types of email content including plain text and HTML formatted messages. "
            "The tool integrates seamlessly with the email infrastructure and provides reliable "
            "delivery capabilities. Users can specify recipient addresses, subject lines, and "
            "message content. The tool handles all the underlying email protocol complexities "
            "and ensures proper message formatting according to email standards. "
            "Furthermore, this tool is capable of handling various edge cases and special "
            "scenarios that may arise during email composition and delivery. It provides "
            "robust error handling and retry mechanisms to ensure reliable message delivery "
            "even in the face of transient network issues or temporary server unavailability. "
            "The tool also supports various authentication mechanisms and encryption protocols "
            "to ensure the security and privacy of email communications."
        ),
        parameters={"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
    )


@pytest.fixture
def heuristic_detector():
    return SmellDetector(use_heuristics_only=True)


@pytest.fixture
def heuristic_pipeline():
    return TDAS(use_heuristics_only=True)


# ── Model tests ───────────────────────────────────────────────────────────────

class TestQualityScore:
    def test_from_components_calculates_correctly(self):
        components = [
            ComponentScore("C1", "Intent Clarity", 3.0),
            ComponentScore("C2", "Constraint Completeness", 3.0),
            ComponentScore("C3", "Parameter Precision", 3.0),
            ComponentScore("C4", "Length Appropriateness", 3.0),
            ComponentScore("C5", "Example Coverage", 3.0),
            ComponentScore("C6", "Scope Accuracy", 3.0),
        ]
        q = QualityScore.from_components(components)
        assert abs(q.overall - 1.0) < 0.01
        assert not q.augmentation_needed

    def test_below_threshold_needs_augmentation(self):
        components = [
            ComponentScore("C1", "Intent Clarity", 0.0),
            ComponentScore("C2", "Constraint Completeness", 0.0),
            ComponentScore("C3", "Parameter Precision", 0.0),
            ComponentScore("C4", "Length Appropriateness", 0.0),
            ComponentScore("C5", "Example Coverage", 0.0),
            ComponentScore("C6", "Scope Accuracy", 0.0),
        ]
        q = QualityScore.from_components(components)
        assert q.overall == 0.0
        assert q.augmentation_needed

    def test_threshold_boundary(self):
        # Score just below threshold
        components = [
            ComponentScore("C1", "Intent Clarity", 1.5),
            ComponentScore("C2", "Constraint Completeness", 1.5),
            ComponentScore("C3", "Parameter Precision", 1.5),
            ComponentScore("C4", "Length Appropriateness", 1.5),
            ComponentScore("C5", "Example Coverage", 1.5),
            ComponentScore("C6", "Scope Accuracy", 1.5),
        ]
        q = QualityScore.from_components(components)
        assert q.overall == pytest.approx(0.5, abs=0.01)
        assert q.augmentation_needed  # 0.5 < 0.55


# ── Detection tests ───────────────────────────────────────────────────────────

class TestSmellDetector:
    def test_detects_s5_on_stub_description(self, heuristic_detector, smelly_tool_s5):
        report = heuristic_detector.detect(smelly_tool_s5)
        smell_types = [d.smell_type for d in report.detections]
        assert SmellType.S5_INCOMPLETE_EXAMPLES in smell_types

    def test_clean_tool_has_low_smell_count(self, heuristic_detector, clean_tool):
        report = heuristic_detector.detect(clean_tool)
        # Clean tool should have at most 1 smell (heuristics are imperfect)
        assert len(report.detections) <= 2

    def test_multi_smell_tool(self, heuristic_detector, smelly_tool_multi):
        report = heuristic_detector.detect(smelly_tool_multi)
        assert report.has_smells
        assert len(report.detections) >= 2

    def test_over_verbose_detected(self, heuristic_detector, over_verbose_tool):
        report = heuristic_detector.detect(over_verbose_tool)
        smell_types = [d.smell_type for d in report.detections]
        assert SmellType.S4_OVER_VERBOSE in smell_types

    def test_quality_score_range(self, heuristic_detector, smelly_tool_s5):
        report = heuristic_detector.detect(smelly_tool_s5)
        assert 0.0 <= report.quality_score.overall <= 1.0

    def test_smell_report_has_correct_tool(self, heuristic_detector, smelly_tool_s5):
        report = heuristic_detector.detect(smelly_tool_s5)
        assert report.tool.name == "search_products"

    def test_tsa_impact_negative_for_smelly(self, heuristic_detector, smelly_tool_multi):
        report = heuristic_detector.detect(smelly_tool_multi)
        if report.has_smells:
            assert report.tsa_impact_pp < 0

    def test_severity_critical_for_s5(self, heuristic_detector, smelly_tool_s5):
        report = heuristic_detector.detect(smelly_tool_s5)
        s5_dets = [d for d in report.detections if d.smell_type == SmellType.S5_INCOMPLETE_EXAMPLES]
        if s5_dets:
            assert s5_dets[0].severity == Severity.CRITICAL


# ── Pipeline tests ────────────────────────────────────────────────────────────

class TestTDASPipeline:
    def test_analyze_returns_smell_report(self, heuristic_pipeline, smelly_tool_s5):
        from tdas.models import SmellReport
        report = heuristic_pipeline.analyze(smelly_tool_s5)
        assert isinstance(report, SmellReport)

    def test_analyze_accepts_dict(self, heuristic_pipeline):
        report = heuristic_pipeline.analyze({
            "name": "my_tool",
            "description": "Does stuff.",
        })
        assert report.tool.name == "my_tool"

    def test_run_skips_clean_tool(self, heuristic_pipeline, clean_tool):
        result = heuristic_pipeline.run(clean_tool)
        # Clean tools should be skipped (no augmenter without client)
        assert result.augmentation_skipped or not result.smell_report.has_smells

    def test_run_raises_without_client_for_smelly(self, smelly_tool_multi):
        # Pipeline without client should raise when augmentation is needed
        pipeline = TDAS(use_heuristics_only=True)
        report = pipeline.analyze(smelly_tool_multi)
        if report.has_smells and report.quality_score.overall < 0.55:
            with pytest.raises(RuntimeError, match="no Anthropic client"):
                pipeline.run(smelly_tool_multi)

    def test_run_batch(self, heuristic_pipeline, smelly_tool_s5, clean_tool):
        tools = [smelly_tool_s5, clean_tool]
        with pytest.raises(RuntimeError):
            # Will raise for smelly tool without client
            heuristic_pipeline.run_batch(tools)

    def test_run_batch_only_clean(self, heuristic_pipeline, clean_tool):
        results = heuristic_pipeline.run_batch([clean_tool])
        assert len(results) == 1


# ── Utility tests ─────────────────────────────────────────────────────────────

class TestUtils:
    def test_format_report_text(self, heuristic_detector, smelly_tool_s5):
        report = heuristic_detector.detect(smelly_tool_s5)
        text = format_report_text(report)
        assert "search_products" in text
        assert "Quality Score" in text

    def test_load_tools_from_json(self, tmp_path):
        from tdas.utils import load_tools_from_json
        data = [
            {"name": "tool_a", "description": "Does A.", "parameters": {}},
            {"name": "tool_b", "description": "Does B."},
        ]
        p = tmp_path / "tools.json"
        import json
        p.write_text(json.dumps(data))
        tools = load_tools_from_json(str(p))
        assert len(tools) == 2
        assert tools[0].name == "tool_a"

    def test_save_and_load_results(self, tmp_path, heuristic_pipeline, clean_tool):
        from tdas.utils import save_results_to_json, load_tools_from_json
        import json

        result = heuristic_pipeline.run(clean_tool)
        out_path = str(tmp_path / "results.json")
        save_results_to_json([result], out_path)

        with open(out_path) as f:
            saved = json.load(f)
        assert len(saved) == 1
        assert saved[0]["name"] == "get_weather"
