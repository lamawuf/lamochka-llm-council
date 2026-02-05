"""Pydantic schemas for Council data structures."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class RoleAssignment(BaseModel):
    """Assignment of a role to a model."""
    role: str = Field(..., description="Role name (chairman, critic, researcher, optimizer)")
    model: str = Field(..., description="Model shorthand or full ID")
    provider: Optional[str] = Field(None, description="Provider override")


class CouncilConfig(BaseModel):
    """Configuration for a council session."""
    roles: List[RoleAssignment] = Field(
        default_factory=lambda: [
            RoleAssignment(role="chairman", model="claude"),  # Best at synthesis
            RoleAssignment(role="critic", model="grok"),      # Sharp critical analysis
            RoleAssignment(role="researcher", model="gemini"), # Good at research
            RoleAssignment(role="optimizer", model="gpt4"),   # Practical improvements
        ]
    )
    max_debate_rounds: int = Field(default=3, ge=1, le=5)
    disagreement_threshold: float = Field(default=7.0, ge=1.0, le=10.0)
    require_hitl: bool = Field(default=False, description="Require human-in-the-loop after Stage 2")
    enable_web_search: bool = Field(default=True, description="Enable web search for Researcher")
    manual_mode: bool = Field(default=False, description="Manual input mode (no API calls)")
    output_format: str = Field(default="markdown", pattern="^(markdown|json)$")

    # Validation settings
    validate_responses: bool = Field(default=True, description="Validate Stage 1 responses have all required twists")
    required_twists: int = Field(default=3, ge=0, description="Number of twists required from each participant (0 to disable)")

    # Synthesis settings
    skip_synthesis: bool = Field(default=False, description="Skip LLM synthesis (Stage 3) - return raw data for external Chairman")


class StageResult(BaseModel):
    """Result from a single stage."""
    stage: int
    stage_name: str
    responses: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class CouncilResult(BaseModel):
    """Complete result from a council session."""
    prompt: str
    config: CouncilConfig
    stages: List[StageResult] = Field(default_factory=list)
    final_answer: str = ""
    disagreement_scores: List[float] = Field(default_factory=list)
    debate_rounds: int = 0
    human_edits: List[str] = Field(default_factory=list)
    total_tokens: int = 0
    duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_markdown(self) -> str:
        """Format result as Markdown."""
        lines = [
            f"# LLM Council Result",
            f"",
            f"**Prompt:** {self.prompt}",
            f"",
            f"**Timestamp:** {self.timestamp.isoformat()}",
            f"**Debate Rounds:** {self.debate_rounds}",
            f"**Total Tokens:** {self.total_tokens}",
            f"",
            "---",
            "",
        ]

        for stage in self.stages:
            lines.append(f"## {stage.stage_name}")
            lines.append("")
            for role, response in stage.responses.items():
                lines.append(f"### {role}")
                lines.append("")
                lines.append(response)
                lines.append("")

        lines.extend([
            "---",
            "",
            "## Final Answer",
            "",
            self.final_answer,
        ])

        return "\n".join(lines)


class ManualResponse(BaseModel):
    """Manual response input for manual mode."""
    role: str
    content: str
