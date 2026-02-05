"""
Core LLM Council logic.
Implements the three-stage council methodology:
1. First Opinions - Initial responses from all models (with validation & retry)
2. Review & Debate - Anonymous cross-review with role-specific perspectives
3. Final Synthesis - Optional LLM synthesis OR raw data for external Chairman

Key improvements:
- Validates that each participant provides ALL requested twists
- Retries participants who don't complete the assignment
- Supports --skip-synthesis for external Chairman (e.g., Claude Code)
"""

import asyncio
import json
import re
import time
from typing import Dict, List, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import (
    DEFAULT_ROLES,
    STAGE_PROMPTS,
    DISAGREEMENT_THRESHOLD,
    MAX_DEBATE_ROUNDS,
)
from providers import ProviderFactory, LLMResponse
from schemas import CouncilConfig, StageResult, CouncilResult, RoleAssignment
from search import web_searcher

console = Console()


# =============================================================================
# Response Validation
# =============================================================================

def count_twists_in_response(response: str, expected_count: int = 3) -> int:
    """Count how many twists/proposals are in a response.

    Looks for patterns like:
    - "Twist 1:", "Twist 2:", "Twist 3:"
    - "### Twist 1", "### Twist 2"
    - "**Twist 1:**", "**Twist 2:**"
    - "1. **Meme Twist**", "2. **Stakes Twist**"
    """
    patterns = [
        r'(?i)twist\s*[#\d:]+',           # Twist 1:, Twist #1, etc.
        r'(?i)###?\s*twist',               # ### Twist, ## Twist
        r'(?i)\*\*twist\s*\d+',            # **Twist 1
        r'(?i)(?:meme|stakes|community)\s*twist',  # Specific twist names
        r'(?i)«.*?twist.*?»',              # «The Meme Twist»
    ]

    found_twists = set()
    for pattern in patterns:
        matches = re.findall(pattern, response)
        found_twists.update(matches)

    # Also check for numbered sections with twist-like content
    numbered_sections = re.findall(r'(?m)^#{1,3}\s*(?:Twist\s*)?\d+[.:]', response)

    return max(len(found_twists), len(numbered_sections))


def validate_stage1_response(response: str, required_twists: int = 3) -> Tuple[bool, int, str]:
    """Validate that a Stage 1 response contains all required twists.

    Returns:
        Tuple of (is_valid, twist_count, error_message)
    """
    if "[Error:" in response:
        return False, 0, "API error"

    twist_count = count_twists_in_response(response, required_twists)

    if twist_count == 0:
        return False, 0, "No twists found - gave analysis instead of proposals"
    elif twist_count < required_twists:
        return False, twist_count, f"Only {twist_count}/{required_twists} twists provided"

    return True, twist_count, "OK"


def get_retry_prompt(original_prompt: str, twist_count: int, required_twists: int) -> str:
    """Generate a retry prompt for incomplete responses."""
    if twist_count == 0:
        return f"""You provided analysis but did NOT complete the assignment.

REQUIRED: Give {required_twists} concrete TWISTS in the format from the original prompt.

Each twist MUST have:
1. Суть изменения (what changes in gameplay)
2. Психологический триггер (FOMO, loss aversion, etc.)
3. Виральная механика (how it makes people share)
4. Техническая сложность (1-10)
5. Реализация (how to implement)

DO NOT give general analysis. Give {required_twists} SPECIFIC PROPOSALS NOW.

Original prompt:
{original_prompt}"""
    else:
        missing = required_twists - twist_count
        return f"""You provided {twist_count} out of {required_twists} twists.

ADD {missing} MORE TWISTS in the same format. Each twist needs:
1. Суть изменения
2. Психологический триггер
3. Виральная механика
4. Техническая сложность (1-10)
5. Реализация

Original prompt:
{original_prompt}"""


class LLMCouncil:
    """Main LLM Council orchestrator."""

    def __init__(self, config: Optional[CouncilConfig] = None):
        self.config = config or CouncilConfig()
        self.results: List[StageResult] = []
        self.disagreement_scores: List[float] = []
        self.human_edits: List[str] = []
        self.total_tokens = 0

    def _get_role_config(self, role_name: str):
        """Get role configuration."""
        return DEFAULT_ROLES.get(role_name.lower())

    def _get_provider_for_role(self, assignment: RoleAssignment):
        """Get the appropriate provider for a role assignment."""
        if assignment.provider:
            return ProviderFactory.get(assignment.provider)

        # Determine provider from model
        provider_name = ProviderFactory.model_to_provider(assignment.model)
        return ProviderFactory.get(provider_name)

    def _anonymize_responses(self, responses: Dict[str, str]) -> str:
        """Convert responses to anonymous format (A, B, C, etc.)."""
        labels = "ABCDEFGHIJ"
        anonymized = []

        for i, (role, content) in enumerate(responses.items()):
            label = labels[i] if i < len(labels) else f"Response {i+1}"
            anonymized.append(f"**Response {label}:**\n{content}\n")

        return "\n---\n".join(anonymized)

    async def _generate_response(
        self,
        assignment: RoleAssignment,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, int]:
        """Generate a response for a role.

        Returns:
            Tuple of (response_content, tokens_used)
        """
        role_config = self._get_role_config(assignment.role)
        provider = self._get_provider_for_role(assignment)

        # Use role-specific system prompt if not provided
        final_system_prompt = system_prompt or (role_config.system_prompt if role_config else None)

        temperature = role_config.temperature if role_config else 0.7

        response = await provider.generate(
            prompt=prompt,
            system_prompt=final_system_prompt,
            model=assignment.model,
            temperature=temperature,
        )

        if not response.success:
            return f"[Error: {response.error}]", 0

        return response.content, response.tokens_used or 0

    async def _run_stage1(self, prompt: str) -> StageResult:
        """Stage 1: First Opinions - Get initial responses from all council members.

        Includes validation and retry for incomplete responses.
        """
        console.print("\n[bold cyan]Stage 1: First Opinions[/bold cyan]")

        responses = {}
        tasks = []
        required_twists = self.config.required_twists

        # Prepare prompts for each role
        stage_prompt = STAGE_PROMPTS["stage1_first_opinion"].format(prompt=prompt)

        for assignment in self.config.roles:
            # Skip chairman in first round (they synthesize later)
            if assignment.role.lower() == "chairman":
                continue

            role_config = self._get_role_config(assignment.role)

            # Special handling for Researcher - add web search
            if assignment.role.lower() == "researcher" and self.config.enable_web_search:
                console.print(f"  [dim]Searching web for: {prompt[:50]}...[/dim]")
                search_results = await web_searcher.search(prompt, num_results=5)
                search_context = web_searcher.format_results(search_results)
                enhanced_prompt = f"{stage_prompt}\n\nWeb Search Results:\n{search_context}"
            else:
                enhanced_prompt = stage_prompt

            tasks.append((
                assignment,
                enhanced_prompt,
            ))

        # Run all tasks and validate with retry
        for assignment, enhanced_prompt in tasks:
            role_name = assignment.role.capitalize()
            console.print(f"  [cyan]{role_name}[/cyan] thinking...")

            content, tokens = await self._generate_response(assignment, enhanced_prompt)
            self.total_tokens += tokens

            # Validate response if validation is enabled
            if self.config.validate_responses and required_twists > 0:
                is_valid, twist_count, error_msg = validate_stage1_response(content, required_twists)

                if not is_valid and "[Error:" not in content:
                    console.print(f"    [yellow]⚠ {role_name}: {error_msg}. Retrying...[/yellow]")

                    # Retry with explicit instructions
                    retry_prompt = get_retry_prompt(prompt, twist_count, required_twists)
                    retry_content, retry_tokens = await self._generate_response(assignment, retry_prompt)
                    self.total_tokens += retry_tokens

                    # Combine original + retry if retry added new content
                    if twist_count > 0:
                        content = f"{content}\n\n---\n**[Additional twists from retry]**\n\n{retry_content}"
                    else:
                        content = retry_content

                    # Re-validate
                    is_valid, new_count, _ = validate_stage1_response(content, required_twists)
                    if is_valid:
                        console.print(f"    [green]✓ {role_name}: Now has {new_count} twists[/green]")
                    else:
                        console.print(f"    [red]✗ {role_name}: Still incomplete ({new_count}/{required_twists})[/red]")
                else:
                    console.print(f"    [green]✓ {role_name}: {twist_count} twists[/green]")

            responses[role_name] = content

        return StageResult(
            stage=1,
            stage_name="Stage 1: First Opinions",
            responses=responses,
        )

    async def _run_stage2(
        self,
        prompt: str,
        stage1_responses: Dict[str, str],
        round_num: int = 1,
    ) -> StageResult:
        """Stage 2: Review & Debate - Anonymous cross-review."""
        console.print(f"\n[bold yellow]Stage 2: Review & Debate (Round {round_num})[/bold yellow]")

        responses = {}
        anonymized = self._anonymize_responses(stage1_responses)

        for assignment in self.config.roles:
            if assignment.role.lower() == "chairman":
                continue

            role_config = self._get_role_config(assignment.role)
            role_name = assignment.role.capitalize()

            review_prompt = STAGE_PROMPTS["stage2_review"].format(
                prompt=prompt,
                role_name=role_name,
                anonymized_responses=anonymized,
            )

            console.print(f"  [yellow]{role_name}[/yellow] reviewing...")
            content, tokens = await self._generate_response(assignment, review_prompt)
            responses[role_name] = content
            self.total_tokens += tokens

        return StageResult(
            stage=2,
            stage_name=f"Stage 2: Review & Debate (Round {round_num})",
            responses=responses,
            metadata={"round": round_num},
        )

    async def _check_disagreement(self, responses: Dict[str, str]) -> float:
        """Check level of disagreement between responses.

        Returns:
            Disagreement score from 1-10
        """
        # Find a working provider for analysis
        provider = ProviderFactory.get_best_available()
        if not provider:
            return 5.0  # Default to middle score if no provider

        anonymized = self._anonymize_responses(responses)
        check_prompt = STAGE_PROMPTS["disagreement_check"].format(responses=anonymized)

        response = await provider.generate(
            prompt=check_prompt,
            temperature=0.3,
            max_tokens=200,
        )

        if not response.success:
            return 5.0

        try:
            # Parse JSON response
            data = json.loads(response.content)
            score = float(data.get("disagreement_score", 5))
            conflicts = data.get("key_conflicts", [])

            if conflicts:
                console.print(f"  [dim]Key conflicts: {', '.join(conflicts)}[/dim]")

            return score
        except (json.JSONDecodeError, ValueError):
            return 5.0

    async def _run_stage3(
        self,
        prompt: str,
        stage1_result: StageResult,
        stage2_results: List[StageResult],
    ) -> StageResult:
        """Stage 3: Final Synthesis - Chairman arbitrates."""
        console.print("\n[bold green]Stage 3: Final Synthesis[/bold green]")

        # Find chairman assignment
        chairman_assignment = None
        for assignment in self.config.roles:
            if assignment.role.lower() == "chairman":
                chairman_assignment = assignment
                break

        if not chairman_assignment:
            # Default to Claude as chairman (best at synthesis)
            chairman_assignment = RoleAssignment(role="chairman", model="claude", provider="openrouter")

        # Format all previous responses
        stage1_formatted = "\n\n".join([
            f"**{role}:** {content}"
            for role, content in stage1_result.responses.items()
        ])

        stage2_formatted = ""
        for result in stage2_results:
            stage2_formatted += f"\n### {result.stage_name}\n"
            for role, content in result.responses.items():
                stage2_formatted += f"\n**{role}:**\n{content}\n"

        synthesis_prompt = STAGE_PROMPTS["stage3_synthesis"].format(
            prompt=prompt,
            stage1_responses=stage1_formatted,
            stage2_responses=stage2_formatted,
        )

        console.print("  [green]Chairman[/green] synthesizing...")
        content, tokens = await self._generate_response(chairman_assignment, synthesis_prompt)
        self.total_tokens += tokens

        return StageResult(
            stage=3,
            stage_name="Stage 3: Final Synthesis",
            responses={"Chairman": content},
        )

    def _human_in_the_loop(self, current_state: str) -> Optional[str]:
        """Allow human to edit/intervene in the process."""
        console.print("\n[bold magenta]Human-in-the-Loop Checkpoint[/bold magenta]")
        console.print("Current state has been saved. You can:")
        console.print("  1. Press Enter to continue without changes")
        console.print("  2. Type your edits/additions")
        console.print("  3. Type 'skip' to skip remaining debate rounds")

        user_input = input("\n> ").strip()

        if not user_input:
            return None
        elif user_input.lower() == "skip":
            return "SKIP_DEBATES"
        else:
            self.human_edits.append(user_input)
            return user_input

    async def _get_manual_responses(self, stage: str, roles: List[str]) -> Dict[str, str]:
        """Get manual responses from user for manual mode."""
        console.print(f"\n[bold]Manual Mode - {stage}[/bold]")
        console.print("Enter responses for each role (Ctrl+D or empty line to finish):\n")

        responses = {}
        for role in roles:
            console.print(f"[cyan]{role}[/cyan]:")
            lines = []
            while True:
                try:
                    line = input()
                    if not line and lines:
                        break
                    lines.append(line)
                except EOFError:
                    break
            responses[role] = "\n".join(lines)

        return responses

    async def run(self, prompt: str) -> CouncilResult:
        """Run the full council session.

        Args:
            prompt: The question/topic for the council to discuss

        Returns:
            CouncilResult with all stages and final answer
        """
        start_time = time.time()

        console.print(Panel(prompt, title="Council Prompt", border_style="blue"))

        # Manual mode handling
        if self.config.manual_mode:
            return await self._run_manual_mode(prompt)

        # Stage 1: First Opinions
        stage1_result = await self._run_stage1(prompt)
        self.results.append(stage1_result)

        # Display Stage 1 results
        for role, content in stage1_result.responses.items():
            console.print(Panel(content[:500] + "..." if len(content) > 500 else content,
                               title=f"[cyan]{role}[/cyan]"))

        # Stage 2: Review & Debate (with potential multiple rounds)
        stage2_results = []
        current_responses = stage1_result.responses
        debate_round = 0

        for round_num in range(1, self.config.max_debate_rounds + 1):
            stage2_result = await self._run_stage2(prompt, current_responses, round_num)
            stage2_results.append(stage2_result)
            self.results.append(stage2_result)
            debate_round = round_num

            # Check disagreement level
            score = await self._check_disagreement(stage2_result.responses)
            self.disagreement_scores.append(score)
            console.print(f"  [dim]Disagreement score: {score:.1f}/10[/dim]")

            # Human-in-the-loop checkpoint
            if self.config.require_hitl:
                human_input = self._human_in_the_loop(
                    json.dumps(stage2_result.responses, indent=2)
                )
                if human_input == "SKIP_DEBATES":
                    break
                elif human_input:
                    # Add human input to the discussion
                    stage2_result.responses["Human"] = human_input

            # Check if we need another round
            if score < self.config.disagreement_threshold:
                console.print("  [green]✓ Consensus reached[/green]")
                break
            elif round_num < self.config.max_debate_rounds:
                console.print("  [yellow]↻ High disagreement, continuing debate...[/yellow]")
                current_responses = stage2_result.responses

        # Stage 3: Final Synthesis (optional)
        if self.config.skip_synthesis:
            console.print("\n[bold yellow]Stage 3: Skipped (--skip-synthesis)[/bold yellow]")
            console.print("  [dim]Raw data returned for external Chairman synthesis[/dim]")
            final_answer = "[Synthesis skipped - use raw data from stages 1-2]"
        else:
            stage3_result = await self._run_stage3(prompt, stage1_result, stage2_results)
            self.results.append(stage3_result)
            final_answer = stage3_result.responses.get("Chairman", "")

        duration = time.time() - start_time

        # Build final result
        result = CouncilResult(
            prompt=prompt,
            config=self.config,
            stages=self.results,
            final_answer=final_answer,
            disagreement_scores=self.disagreement_scores,
            debate_rounds=debate_round,
            human_edits=self.human_edits,
            total_tokens=self.total_tokens,
            duration_seconds=duration,
        )

        console.print("\n" + "=" * 60)
        console.print(Panel(final_answer, title="[bold green]Final Council Answer[/bold green]"))
        console.print(f"\n[dim]Completed in {duration:.1f}s | {self.total_tokens} tokens | {debate_round} debate round(s)[/dim]")

        return result

    async def _run_manual_mode(self, prompt: str) -> CouncilResult:
        """Run council in manual mode (no API calls)."""
        start_time = time.time()

        roles = [a.role.capitalize() for a in self.config.roles if a.role.lower() != "chairman"]

        # Stage 1
        stage1_responses = await self._get_manual_responses("Stage 1: First Opinions", roles)
        stage1_result = StageResult(
            stage=1,
            stage_name="Stage 1: First Opinions (Manual)",
            responses=stage1_responses,
        )
        self.results.append(stage1_result)

        # Stage 2
        console.print("\n[dim]Anonymized Stage 1 responses for review:[/dim]")
        console.print(self._anonymize_responses(stage1_responses))

        stage2_responses = await self._get_manual_responses("Stage 2: Review & Debate", roles)
        stage2_result = StageResult(
            stage=2,
            stage_name="Stage 2: Review & Debate (Manual)",
            responses=stage2_responses,
        )
        self.results.append(stage2_result)

        # Stage 3
        console.print("\n[bold]Enter Chairman's final synthesis:[/bold]")
        lines = []
        while True:
            try:
                line = input()
                if not line and lines:
                    break
                lines.append(line)
            except EOFError:
                break
        chairman_response = "\n".join(lines)

        stage3_result = StageResult(
            stage=3,
            stage_name="Stage 3: Final Synthesis (Manual)",
            responses={"Chairman": chairman_response},
        )
        self.results.append(stage3_result)

        return CouncilResult(
            prompt=prompt,
            config=self.config,
            stages=self.results,
            final_answer=chairman_response,
            debate_rounds=1,
            duration_seconds=time.time() - start_time,
        )


async def run_council(
    prompt: str,
    config: Optional[CouncilConfig] = None,
) -> CouncilResult:
    """Convenience function to run a council session.

    Args:
        prompt: The question/topic for discussion
        config: Optional configuration

    Returns:
        CouncilResult with all stages and final answer
    """
    council = LLMCouncil(config)
    return await council.run(prompt)
