#!/usr/bin/env python3
"""
Lamochka LLM Council - CLI Entry Point

A multi-LLM council for collaborative AI decision-making with role-based debates.

Usage:
    python main.py "Your question here"
    python main.py --prompt "Your question" --roles custom.json
    python main.py --hitl "Complex question requiring human oversight"
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import settings, DEFAULT_ROLES
from council import LLMCouncil, run_council
from schemas import CouncilConfig, RoleAssignment
from providers import ProviderFactory

app = typer.Typer(
    name="council",
    help="Lamochka LLM Council - Multi-LLM collaborative decision making",
    add_completion=False,
)
console = Console()


def load_roles_file(path: str) -> list[RoleAssignment]:
    """Load custom roles from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    roles = []
    for item in data:
        roles.append(RoleAssignment(
            role=item["role"],
            model=item.get("model", "claude"),
            provider=item.get("provider"),
        ))
    return roles


@app.command()
def run(
    prompt: str = typer.Argument(None, help="The question or topic for the council"),
    prompt_file: Optional[Path] = typer.Option(
        None, "--file", "-f",
        help="Read prompt from a file",
    ),
    roles_file: Optional[Path] = typer.Option(
        None, "--roles", "-r",
        help="Custom roles configuration JSON file",
    ),
    max_rounds: int = typer.Option(
        3, "--max-rounds", "-m",
        help="Maximum number of debate rounds",
        min=1, max=5,
    ),
    disagreement_threshold: float = typer.Option(
        7.0, "--threshold", "-t",
        help="Disagreement threshold (1-10) to trigger additional rounds",
        min=1.0, max=10.0,
    ),
    require_hitl: bool = typer.Option(
        False, "--hitl",
        help="Require human-in-the-loop after Stage 2",
    ),
    manual_mode: bool = typer.Option(
        False, "--manual",
        help="Manual mode - enter responses yourself (no API calls)",
    ),
    no_search: bool = typer.Option(
        False, "--no-search",
        help="Disable web search for Researcher role",
    ),
    output_format: str = typer.Option(
        "markdown", "--output", "-o",
        help="Output format: markdown or json",
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--save", "-s",
        help="Save result to file",
    ),
):
    """
    Run an LLM Council session.

    The council follows three stages:
    1. First Opinions - Each model responds to the prompt
    2. Review & Debate - Models review each other's responses anonymously
    3. Final Synthesis - Chairman synthesizes the final answer

    Examples:
        council run "What is the best programming language for AI?"
        council run --hitl "Should we use microservices or monolith?"
        council run -f prompt.txt --roles custom_roles.json
    """
    # Get prompt from argument or file
    if prompt_file:
        final_prompt = prompt_file.read_text().strip()
    elif prompt:
        final_prompt = prompt
    else:
        console.print("[red]Error: Either prompt or --file is required[/red]")
        raise typer.Exit(1)

    # Build configuration
    config = CouncilConfig(
        max_debate_rounds=max_rounds,
        disagreement_threshold=disagreement_threshold,
        require_hitl=require_hitl,
        manual_mode=manual_mode,
        enable_web_search=not no_search,
        output_format=output_format,
    )

    # Load custom roles if provided
    if roles_file:
        try:
            config.roles = load_roles_file(str(roles_file))
        except Exception as e:
            console.print(f"[red]Error loading roles file: {e}[/red]")
            raise typer.Exit(1)

    # Check provider availability
    if not manual_mode:
        available = ProviderFactory.get_available()
        if not available:
            console.print("[red]Error: No LLM providers available.[/red]")
            console.print("Configure at least one API key in .env file.")
            console.print("Or use --manual for manual input mode.")
            raise typer.Exit(1)

        console.print(f"[dim]Available providers: {', '.join(available.keys())}[/dim]")

    # Run the council
    try:
        result = asyncio.run(run_council(final_prompt, config))

        # Save output if requested
        if output_file:
            if output_format == "json":
                output_file.write_text(result.model_dump_json(indent=2))
            else:
                output_file.write_text(result.to_markdown())
            console.print(f"\n[green]Result saved to {output_file}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Council session interrupted[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status():
    """Check status of available LLM providers."""
    console.print("\n[bold]LLM Provider Status[/bold]\n")

    table = Table(show_header=True)
    table.add_column("Provider")
    table.add_column("Status")
    table.add_column("Default Model")

    providers = [
        ("openrouter", "OpenRouter"),
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic"),
        ("google", "Google"),
        ("xai", "xAI (Grok)"),
        ("ollama", "Ollama (Local)"),
    ]

    for name, display_name in providers:
        try:
            provider = ProviderFactory.get(name)
            if provider.is_available():
                status = "[green]✓ Available[/green]"
                model = getattr(provider, "default_model", "N/A")
            else:
                status = "[red]✗ Not configured[/red]"
                model = "-"
        except Exception:
            status = "[red]✗ Error[/red]"
            model = "-"

        table.add_row(display_name, status, model)

    console.print(table)
    console.print("\n[dim]Configure API keys in .env file[/dim]")


@app.command()
def roles():
    """List available council roles and their configurations."""
    console.print("\n[bold]Council Roles[/bold]\n")

    for name, config in DEFAULT_ROLES.items():
        console.print(Panel(
            f"[bold]{config.name}[/bold]\n"
            f"[dim]Default model:[/dim] {config.default_model}\n"
            f"[dim]Temperature:[/dim] {config.temperature}\n\n"
            f"{config.description}\n\n"
            f"[dim]System prompt preview:[/dim]\n"
            f"[italic]{config.system_prompt[:200]}...[/italic]",
            title=name.upper(),
            border_style="blue",
        ))


@app.command()
def example():
    """Show example usage and configuration."""
    example_config = {
        "roles": [
            {"role": "chairman", "model": "claude"},
            {"role": "critic", "model": "grok"},
            {"role": "researcher", "model": "gemini"},
            {"role": "optimizer", "model": "gpt4"},
        ]
    }

    console.print("\n[bold]Example Custom Roles File (roles.json):[/bold]\n")
    console.print(json.dumps(example_config, indent=2))

    console.print("\n[bold]Example Commands:[/bold]\n")
    examples = [
        ('Basic usage:', 'council run "What is the best approach to solve this problem?"'),
        ('With HITL:', 'council run --hitl "Complex architectural decision"'),
        ('Custom roles:', 'council run -r roles.json "Your question"'),
        ('Save output:', 'council run -o json -s result.json "Your question"'),
        ('Manual mode:', 'council run --manual "Question for manual responses"'),
        ('From file:', 'council run -f prompt.txt'),
    ]

    for desc, cmd in examples:
        console.print(f"  [cyan]{desc}[/cyan]")
        console.print(f"    {cmd}\n")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
):
    """Start the FastAPI server for API access."""
    import uvicorn
    console.print(f"[green]Starting API server at http://{host}:{port}[/green]")
    console.print("[dim]API docs available at /docs[/dim]")
    uvicorn.run("api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
