"""
Configuration for Lamochka LLM Council.
Defines default roles, models, prompts, and settings.
"""

from typing import Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Global env file path (shared across all projects)
GLOBAL_ENV_PATH = Path.home() / ".claude" / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=[GLOBAL_ENV_PATH, ".env"],  # Global first, then local override
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys (support both naming conventions)
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = Field(default=None, alias="gemini_api_key")
    gemini_api_key: Optional[str] = None  # Alias for google_api_key
    xai_api_key: Optional[str] = None
    serper_api_key: Optional[str] = None

    # Ollama
    ollama_host: str = "http://localhost:11434"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Defaults
    default_provider: str = "openrouter"
    log_level: str = "INFO"
    max_retries: int = 3
    request_timeout: int = 120


settings = Settings()


# ==============================================
# Default Model Mappings
# ==============================================

# OpenRouter model IDs
OPENROUTER_MODELS = {
    "claude": "anthropic/claude-3.5-sonnet",
    "gpt4": "openai/gpt-4o",
    "gemini": "google/gemini-pro-1.5",
    "grok": "x-ai/grok-2-1212",
    "llama": "meta-llama/llama-3.1-70b-instruct",
}

# Direct API model IDs
DIRECT_MODELS = {
    "claude": "claude-sonnet-4-20250514",
    "gpt4": "gpt-4o",
    "gemini": "gemini-2.0-flash",
    "grok": "grok-3-latest",
}

# Ollama models (local)
OLLAMA_MODELS = {
    "llama": "llama3.1:70b",
    "mistral": "mistral:latest",
    "mixtral": "mixtral:latest",
}


# ==============================================
# Role Definitions
# ==============================================

class RoleConfig:
    """Configuration for a council role."""

    def __init__(
        self,
        name: str,
        description: str,
        default_model: str,
        system_prompt: str,
        temperature: float = 0.7,
    ):
        self.name = name
        self.description = description
        self.default_model = default_model
        self.system_prompt = system_prompt
        self.temperature = temperature


# Default role configurations
DEFAULT_ROLES = {
    "chairman": RoleConfig(
        name="Chairman",
        description="Arbitrator who synthesizes all opinions and makes final decisions",
        default_model="gpt4",  # Using GPT-4 as Chairman (no Anthropic key)
        system_prompt="""You are the Chairman of an LLM Council - a wise arbitrator and synthesizer.

Your responsibilities:
1. Analyze all council member opinions objectively
2. Identify points of agreement and disagreement
3. Weigh the merits of each argument
4. Synthesize a final, balanced conclusion
5. Explain your reasoning clearly

Be fair, thorough, and diplomatic. Your goal is to find the best answer by combining insights from all perspectives.
When synthesizing, cite which council member's input you're incorporating.""",
        temperature=0.5,
    ),

    "critic": RoleConfig(
        name="Critic",
        description="Finds weaknesses, errors, and biases in arguments",
        default_model="grok",
        system_prompt="""You are the Critic of an LLM Council - a rigorous devil's advocate.

Your responsibilities:
1. Identify logical fallacies and weak arguments
2. Point out potential biases and blind spots
3. Challenge assumptions and conventional thinking
4. Suggest counterarguments and alternative viewpoints
5. Be constructively critical - aim to improve, not destroy

Be incisive and unafraid to disagree. Your skepticism makes the council stronger.
Always explain WHY something is problematic and suggest how to fix it.""",
        temperature=0.8,
    ),

    "researcher": RoleConfig(
        name="Researcher",
        description="Fact-checker with web search capabilities",
        default_model="gemini",
        system_prompt="""You are the Researcher of an LLM Council - a meticulous fact-checker.

Your responsibilities:
1. Verify factual claims with evidence
2. Find relevant data and statistics
3. Cite sources and provide context
4. Identify when claims need verification
5. Distinguish between facts, opinions, and speculation

You have access to web search. Use it to verify claims and find supporting evidence.
Always cite your sources. If you can't verify something, say so explicitly.""",
        temperature=0.3,
    ),

    "optimizer": RoleConfig(
        name="Optimizer",
        description="Improves and refines ideas for practicality",
        default_model="gpt4",
        system_prompt="""You are the Optimizer of an LLM Council - a practical improver of ideas.

Your responsibilities:
1. Take ideas and make them more actionable
2. Identify implementation challenges and solutions
3. Suggest improvements and refinements
4. Consider real-world constraints and trade-offs
5. Propose step-by-step action plans

Focus on HOW to make things work. Be practical, specific, and solution-oriented.
Transform abstract ideas into concrete, implementable plans.""",
        temperature=0.6,
    ),
}


# ==============================================
# Stage Prompts
# ==============================================

STAGE_PROMPTS = {
    "stage1_first_opinion": """
Provide your initial response to the following question/prompt.
Be thorough and share your genuine perspective.

PROMPT:
{prompt}

Respond with your analysis, reasoning, and conclusions.
""",

    "stage2_review": """
You are reviewing responses from other council members (anonymized as A, B, C, etc.).
Your role: {role_name}

ORIGINAL PROMPT:
{prompt}

RESPONSES TO REVIEW:
{anonymized_responses}

Based on your role as {role_name}:
1. Analyze each response's strengths and weaknesses
2. Identify agreements and disagreements
3. Apply your role's perspective (criticism, research, optimization, etc.)
4. Provide your refined opinion

Be specific in your critique. Reference responses by letter (A, B, etc.).
""",

    "stage3_synthesis": """
As Chairman, synthesize the council's discussion into a final answer.

ORIGINAL PROMPT:
{prompt}

STAGE 1 - INITIAL OPINIONS:
{stage1_responses}

STAGE 2 - REVIEWS AND DEBATES:
{stage2_responses}

Your task:
1. Identify the key insights from all contributions
2. Resolve disagreements fairly
3. Synthesize a comprehensive final answer
4. Explain your reasoning and which inputs influenced your conclusion

Provide the definitive council answer.
""",

    "disagreement_check": """
Analyze these responses and rate the level of disagreement on a scale of 1-10.

RESPONSES:
{responses}

Consider:
- Do they reach similar conclusions?
- Are the reasoning approaches aligned?
- Are there fundamental conflicts?

Respond with ONLY a JSON object:
{{"disagreement_score": <1-10>, "key_conflicts": ["conflict1", "conflict2"]}}
""",
}


# ==============================================
# Thresholds and Limits
# ==============================================

# If disagreement score is above this, trigger additional debate round
DISAGREEMENT_THRESHOLD = 7

# Maximum number of debate rounds
MAX_DEBATE_ROUNDS = 3

# Maximum tokens per response
MAX_TOKENS = 4096

# Rate limiting (requests per minute per provider)
RATE_LIMITS = {
    "openrouter": 60,
    "openai": 60,
    "anthropic": 50,
    "google": 60,
    "xai": 30,
    "ollama": 1000,  # Local, no real limit
}
