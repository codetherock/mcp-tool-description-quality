"""
Data models for TDAS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SmellType(str, Enum):
    """The six MCP tool description smell categories (S1-S6)."""

    S1_VAGUE_INTENT = "S1_vague_intent"
    S2_MISSING_CONSTRAINTS = "S2_missing_constraints"
    S3_AMBIGUOUS_PARAMETERS = "S3_ambiguous_parameters"
    S4_OVER_VERBOSE = "S4_over_verbose"
    S5_INCOMPLETE_EXAMPLES = "S5_incomplete_examples"
    S6_MISLEADING_SCOPE = "S6_misleading_scope"


SHORT_SMELL_CODES: dict[str, SmellType] = {
    "S1": SmellType.S1_VAGUE_INTENT,
    "S2": SmellType.S2_MISSING_CONSTRAINTS,
    "S3": SmellType.S3_AMBIGUOUS_PARAMETERS,
    "S4": SmellType.S4_OVER_VERBOSE,
    "S5": SmellType.S5_INCOMPLETE_EXAMPLES,
    "S6": SmellType.S6_MISLEADING_SCOPE,
}


class Severity(str, Enum):
    """Smell severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Metadata about each smell type
SMELL_METADATA: dict[SmellType, dict] = {
    SmellType.S1_VAGUE_INTENT: {
        "name": "Vague Intent",
        "severity": Severity.HIGH,
        "tsa_impact_pp": -26.1,
        "prevalence": 0.613,
        "description": "Tool purpose stated too broadly; missing domain scope.",
        "fix": "Clarify the primary function and target domain unambiguously.",
    },
    SmellType.S2_MISSING_CONSTRAINTS: {
        "name": "Missing Constraints",
        "severity": Severity.HIGH,
        "tsa_impact_pp": -23.4,
        "prevalence": 0.547,
        "description": "Absent pre/post-conditions, auth requirements, or rate limits.",
        "fix": "Document all preconditions, authentication requirements, rate limits, and failure modes.",
    },
    SmellType.S3_AMBIGUOUS_PARAMETERS: {
        "name": "Ambiguous Parameters",
        "severity": Severity.MEDIUM,
        "tsa_impact_pp": -19.7,
        "prevalence": 0.482,
        "description": "Parameter names or types underspecified; no format examples.",
        "fix": "Add type annotations, format examples, and semantic descriptions for all parameters.",
    },
    SmellType.S4_OVER_VERBOSE: {
        "name": "Over-verbose",
        "severity": Severity.MEDIUM,
        "tsa_impact_pp": -11.2,
        "prevalence": 0.396,
        "description": "Description exceeds 200 tokens; irrelevant prose inflates context.",
        "fix": "Trim to essential information only; target 50-150 tokens.",
    },
    SmellType.S5_INCOMPLETE_EXAMPLES: {
        "name": "Incomplete Examples",
        "severity": Severity.CRITICAL,
        "tsa_impact_pp": -30.5,
        "prevalence": 0.724,
        "description": "No I/O examples; agent cannot infer usage from description alone.",
        "fix": "Add at least one concrete input/output example showing typical tool usage.",
    },
    SmellType.S6_MISLEADING_SCOPE: {
        "name": "Misleading Scope",
        "severity": Severity.HIGH,
        "tsa_impact_pp": -18.3,
        "prevalence": 0.289,
        "description": "Description implies broader capability than actual implementation.",
        "fix": "Explicitly bound what the tool can and cannot do.",
    },
}


@dataclass
class ToolDescription:
    """Represents a single MCP tool description."""

    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    server_name: Optional[str] = None
    server_url: Optional[str] = None
    server_prefix: Optional[str] = None
    domain: Optional[str] = None
    smells_expected: list[str] = field(default_factory=list)
    quality_notes: Optional[str] = None

    def token_count(self) -> int:
        """Rough token count (words * 1.3)."""
        return int(len(self.description.split()) * 1.3)


@dataclass
class ComponentScore:
    """Score for one of the six quality components (0-3 scale)."""

    component: str  # C1-C6
    label: str
    score: float  # 0.0 - 3.0
    rationale: str = ""

    @property
    def normalized(self) -> float:
        return self.score / 3.0


@dataclass
class QualityScore:
    """
    Overall quality score Q(d) ∈ [0, 1].
    Computed as weighted sum of six component scores.
    Q < 0.55 triggers augmentation.
    """

    overall: float
    components: list[ComponentScore] = field(default_factory=list)
    augmentation_needed: bool = False

    THRESHOLD = 0.55

    # Regression-style weights for the six quality components.
    WEIGHTS = {
        "C1": 0.22,  # Intent Clarity
        "C2": 0.18,  # Constraint Completeness
        "C3": 0.17,  # Parameter Precision
        "C4": 0.10,  # Length Appropriateness
        "C5": 0.21,  # Example Coverage
        "C6": 0.12,  # Scope Accuracy
    }

    @classmethod
    def from_components(cls, components: list[ComponentScore]) -> "QualityScore":
        weights = cls.WEIGHTS
        total = sum(weights.get(c.component, 0) * c.normalized for c in components)
        return cls(
            overall=round(total, 4),
            components=components,
            augmentation_needed=total < cls.THRESHOLD,
        )


@dataclass
class SmellDetection:
    """A single detected smell with supporting evidence."""

    smell_type: SmellType
    severity: Severity
    confidence: float  # 0.0 - 1.0
    evidence: str  # Why this smell was detected
    component_score: float  # The C-score that triggered it (0-3)
    threshold: float  # The threshold it fell below


@dataclass
class SmellReport:
    """Full smell analysis report for one tool description."""

    tool: ToolDescription
    quality_score: QualityScore
    detections: list[SmellDetection] = field(default_factory=list)
    smell_set: list[SmellType] = field(default_factory=list)

    @property
    def has_smells(self) -> bool:
        return len(self.detections) > 0

    @property
    def max_severity(self) -> Optional[Severity]:
        if not self.detections:
            return None
        order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]
        for sev in order:
            if any(d.severity == sev for d in self.detections):
                return sev
        return None

    @property
    def tsa_impact_pp(self) -> float:
        """Estimated total TSA impact in percentage points."""
        if not self.detections:
            return 0.0
        return min(
            sum(SMELL_METADATA[d.smell_type]["tsa_impact_pp"] for d in self.detections), -50.0
        )  # floor at -50pp


@dataclass
class AugmentationResult:
    """Result of augmenting a tool description."""

    original: ToolDescription
    augmented: ToolDescription
    smell_report: SmellReport
    augmented_quality: QualityScore
    smells_addressed: list[SmellType] = field(default_factory=list)
    token_delta: int = 0
    augmentation_skipped: bool = False
    skip_reason: str = ""

    @property
    def quality_delta(self) -> float:
        return self.augmented_quality.overall - self.smell_report.quality_score.overall
