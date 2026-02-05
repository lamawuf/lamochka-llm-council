"""
FastAPI server for LLM Council.
Provides REST API for programmatic access and future UI.
"""

from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings, DEFAULT_ROLES
from council import run_council
from schemas import CouncilConfig, CouncilResult, RoleAssignment
from providers import ProviderFactory

app = FastAPI(
    title="Lamochka LLM Council API",
    description="Multi-LLM Council for collaborative AI decision-making",
    version="1.0.0",
)

# CORS for future UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory storage for async results
_results: dict[str, CouncilResult] = {}
_pending: set[str] = set()


class CouncilRequest(BaseModel):
    """Request to run a council session."""
    prompt: str = Field(..., min_length=1, description="The question or topic")
    roles: Optional[list[RoleAssignment]] = None
    max_debate_rounds: int = Field(default=3, ge=1, le=5)
    disagreement_threshold: float = Field(default=7.0, ge=1.0, le=10.0)
    enable_web_search: bool = True
    output_format: str = Field(default="markdown", pattern="^(markdown|json)$")


class AsyncCouncilResponse(BaseModel):
    """Response for async council request."""
    session_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    """Provider status response."""
    provider: str
    available: bool
    default_model: Optional[str] = None


@app.get("/")
async def root():
    """API root - health check."""
    return {
        "name": "Lamochka LLM Council",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    available_providers = ProviderFactory.get_available()
    return {
        "status": "healthy" if available_providers else "degraded",
        "providers_available": len(available_providers),
    }


@app.get("/providers", response_model=list[StatusResponse])
async def list_providers():
    """List all providers and their status."""
    providers = []
    for name in ["openrouter", "openai", "anthropic", "google", "xai", "ollama"]:
        try:
            p = ProviderFactory.get(name)
            providers.append(StatusResponse(
                provider=name,
                available=p.is_available(),
                default_model=getattr(p, "default_model", None),
            ))
        except Exception:
            providers.append(StatusResponse(
                provider=name,
                available=False,
            ))
    return providers


@app.get("/roles")
async def list_roles():
    """List available council roles."""
    return {
        name: {
            "name": config.name,
            "description": config.description,
            "default_model": config.default_model,
            "temperature": config.temperature,
        }
        for name, config in DEFAULT_ROLES.items()
    }


@app.post("/council/run")
async def council_run(request: CouncilRequest) -> CouncilResult:
    """
    Run a synchronous council session.

    This endpoint blocks until the council completes.
    For long-running sessions, use /council/start for async execution.
    """
    config = CouncilConfig(
        max_debate_rounds=request.max_debate_rounds,
        disagreement_threshold=request.disagreement_threshold,
        enable_web_search=request.enable_web_search,
        output_format=request.output_format,
    )

    if request.roles:
        config.roles = request.roles

    # Check providers
    if not ProviderFactory.get_available():
        raise HTTPException(
            status_code=503,
            detail="No LLM providers available. Configure API keys.",
        )

    try:
        result = await run_council(request.prompt, config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/council/start", response_model=AsyncCouncilResponse)
async def council_start(
    request: CouncilRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start an async council session.

    Returns a session_id to poll for results.
    """
    import uuid

    session_id = str(uuid.uuid4())
    _pending.add(session_id)

    async def _run_session():
        config = CouncilConfig(
            max_debate_rounds=request.max_debate_rounds,
            disagreement_threshold=request.disagreement_threshold,
            enable_web_search=request.enable_web_search,
            output_format=request.output_format,
        )
        if request.roles:
            config.roles = request.roles

        try:
            result = await run_council(request.prompt, config)
            _results[session_id] = result
        finally:
            _pending.discard(session_id)

    background_tasks.add_task(_run_session)

    return AsyncCouncilResponse(
        session_id=session_id,
        status="pending",
        message="Council session started. Poll /council/status/{session_id} for results.",
    )


@app.get("/council/status/{session_id}")
async def council_status(session_id: str):
    """Check status of an async council session."""
    if session_id in _pending:
        return {"session_id": session_id, "status": "running"}

    if session_id in _results:
        return {
            "session_id": session_id,
            "status": "completed",
            "result": _results[session_id],
        }

    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/council/result/{session_id}")
async def council_result(session_id: str) -> CouncilResult:
    """Get the result of a completed council session."""
    if session_id in _pending:
        raise HTTPException(status_code=202, detail="Session still running")

    if session_id not in _results:
        raise HTTPException(status_code=404, detail="Session not found")

    return _results[session_id]
