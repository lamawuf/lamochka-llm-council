#!/usr/bin/env python3
"""
Security checker for GitHub repositories.
Scans for drainers, backdoors, hardcoded secrets, and suspicious code.
"""

import os
import re
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


@dataclass
class Finding:
    """A security finding."""
    severity: str  # HIGH, MEDIUM, LOW
    category: str
    description: str
    file: str
    line: Optional[int] = None
    code: Optional[str] = None


@dataclass
class ScanResult:
    """Results of a security scan."""
    repo_url: str
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0

    @property
    def is_safe(self) -> bool:
        return not any(f.severity == "HIGH" for f in self.findings)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "HIGH")

    @property
    def medium_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "MEDIUM")

    @property
    def low_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "LOW")


class RepoSecurityChecker:
    """Scans repositories for security issues."""

    # Patterns to search for
    PATTERNS = {
        # HIGH severity - likely malicious
        "HIGH": {
            "obfuscated_python": {
                "pattern": r"(exec|eval)\s*\(\s*(base64\.b64decode|codecs\.decode|compile)\s*\(",
                "description": "Obfuscated code execution (likely backdoor)",
                "extensions": [".py"],
            },
            "obfuscated_js": {
                "pattern": r"eval\s*\(\s*(atob|String\.fromCharCode)\s*\(",
                "description": "Obfuscated JavaScript (likely drainer)",
                "extensions": [".js", ".ts"],
            },
            "wallet_theft": {
                "pattern": r"(private_key|seed_phrase|mnemonic|secret_key)\s*[=:]\s*['\"]?[a-zA-Z0-9]",
                "description": "Hardcoded wallet credentials",
                "extensions": [".py", ".js", ".ts", ".json"],
            },
            "eth_private_key": {
                "pattern": r"0x[a-fA-F0-9]{64}",
                "description": "Hardcoded Ethereum private key",
                "extensions": [".py", ".js", ".ts", ".json", ".env"],
            },
            "curl_pipe_bash": {
                "pattern": r"curl\s+.*\|\s*(ba)?sh|wget\s+.*\|\s*(ba)?sh",
                "description": "Remote code execution via curl|bash",
                "extensions": [".sh", ".py", ".js"],
            },
            "data_exfil": {
                "pattern": r"(requests\.(post|get)|fetch|axios)\s*\(['\"]https?://(?!api\.(openai|anthropic|google|x\.ai|openrouter))",
                "description": "Data exfiltration to unknown server",
                "extensions": [".py", ".js", ".ts"],
            },
        },
        # MEDIUM severity - suspicious
        "MEDIUM": {
            "api_key_hardcoded": {
                "pattern": r"(api_key|apikey|api_secret)\s*[=:]\s*['\"][a-zA-Z0-9_\-]{20,}['\"]",
                "description": "Hardcoded API key",
                "extensions": [".py", ".js", ".ts"],
            },
            "openai_key": {
                "pattern": r"sk-[a-zA-Z0-9]{32,}",
                "description": "Hardcoded OpenAI API key",
                "extensions": [".py", ".js", ".ts", ".json"],
            },
            "base64_decode": {
                "pattern": r"base64\.(b64decode|decode)|atob\(",
                "description": "Base64 decoding (check context)",
                "extensions": [".py", ".js", ".ts"],
            },
            "subprocess_shell": {
                "pattern": r"subprocess\.(run|call|Popen)\s*\([^)]*shell\s*=\s*True",
                "description": "Shell command execution",
                "extensions": [".py"],
            },
            "eth_transaction": {
                "pattern": r"eth_sendTransaction|signTransaction|sendRawTransaction",
                "description": "Ethereum transaction (verify intent)",
                "extensions": [".py", ".js", ".ts"],
            },
        },
        # LOW severity - worth noting
        "LOW": {
            "env_file_tracked": {
                "pattern": r"^\.env$",
                "description": ".env file in repository",
                "extensions": [],  # Special handling
            },
            "exec_eval": {
                "pattern": r"\b(exec|eval)\s*\(",
                "description": "Dynamic code execution",
                "extensions": [".py", ".js"],
            },
            "rm_rf": {
                "pattern": r"rm\s+-rf\s+/|shutil\.rmtree\s*\(['\"]\/",
                "description": "Dangerous file deletion",
                "extensions": [".py", ".sh"],
            },
        },
    }

    # Known malicious package names (typosquatting)
    MALICIOUS_PACKAGES = {
        "python": [
            "python-binance" if "binanace" in "x" else None,  # Example pattern
            "colourama",  # vs colorama
            "python-mysql",  # vs mysql-connector-python
            "djanga",  # vs django
            "flaask",  # vs flask
            "reqeusts",  # vs requests
            "urlib3",  # vs urllib3
            "beatuifulsoup",  # vs beautifulsoup4
            "crytpography",  # vs cryptography
        ],
        "npm": [
            "electorn",  # vs electron
            "crossenv",  # vs cross-env (known malware)
            "mongose",  # vs mongoose
            "lodashs",  # vs lodash
        ],
    }

    def __init__(self):
        self.temp_dir: Optional[Path] = None

    def clone_repo(self, repo_url: str) -> Path:
        """Clone repository to temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="security_check_"))

        console.print(f"[dim]Cloning {repo_url}...[/dim]")

        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(self.temp_dir / "repo")],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone: {result.stderr}")

        return self.temp_dir / "repo"

    def cleanup(self):
        """Remove temporary directory."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def scan_file(self, filepath: Path, severity: str, checks: dict) -> list[Finding]:
        """Scan a single file for patterns."""
        findings = []

        try:
            content = filepath.read_text(errors="ignore")
            lines = content.split("\n")

            for check_name, check_config in checks.items():
                extensions = check_config.get("extensions", [])

                # Skip if extension doesn't match
                if extensions and filepath.suffix not in extensions:
                    continue

                pattern = check_config["pattern"]

                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        # Skip if it's in a comment or example
                        if self._is_false_positive(line, filepath):
                            continue

                        findings.append(Finding(
                            severity=severity,
                            category=check_name,
                            description=check_config["description"],
                            file=str(filepath),
                            line=i,
                            code=line.strip()[:100],
                        ))
        except Exception:
            pass  # Skip unreadable files

        return findings

    def _is_false_positive(self, line: str, filepath: Path) -> bool:
        """Check if finding is likely a false positive."""
        line_lower = line.lower().strip()

        # Skip comments
        if line_lower.startswith("#") or line_lower.startswith("//"):
            return True

        # Skip example/placeholder values
        false_positive_indicators = [
            "example", "xxx", "your_", "your-", "<your",
            "placeholder", "dummy", "test", "fake",
            ".example", "sample", "template"
        ]

        if any(ind in line_lower for ind in false_positive_indicators):
            return True

        # Skip documentation files
        if filepath.suffix in [".md", ".rst", ".txt"]:
            return True

        return False

    def check_dependencies(self, repo_path: Path) -> list[Finding]:
        """Check for suspicious dependencies."""
        findings = []

        # Check Python requirements
        for req_file in ["requirements.txt", "requirements-dev.txt", "setup.py", "pyproject.toml"]:
            req_path = repo_path / req_file
            if req_path.exists():
                content = req_path.read_text(errors="ignore").lower()
                for pkg in self.MALICIOUS_PACKAGES["python"]:
                    if pkg and pkg in content:
                        findings.append(Finding(
                            severity="HIGH",
                            category="malicious_package",
                            description=f"Potentially malicious package: {pkg}",
                            file=str(req_path),
                        ))

        # Check npm packages
        package_json = repo_path / "package.json"
        if package_json.exists():
            content = package_json.read_text(errors="ignore").lower()
            for pkg in self.MALICIOUS_PACKAGES["npm"]:
                if pkg and pkg in content:
                    findings.append(Finding(
                        severity="HIGH",
                        category="malicious_package",
                        description=f"Potentially malicious npm package: {pkg}",
                        file=str(package_json),
                    ))

        return findings

    def check_gitignore(self, repo_path: Path) -> list[Finding]:
        """Check if sensitive files are properly ignored."""
        findings = []

        gitignore = repo_path / ".gitignore"
        gitignore_content = ""

        if gitignore.exists():
            gitignore_content = gitignore.read_text(errors="ignore")

        # Check if .env is ignored
        if ".env" not in gitignore_content:
            findings.append(Finding(
                severity="MEDIUM",
                category="env_not_ignored",
                description=".env not in .gitignore (secrets may leak)",
                file=".gitignore",
            ))

        # Check if actual .env exists in repo
        if (repo_path / ".env").exists():
            findings.append(Finding(
                severity="HIGH",
                category="env_in_repo",
                description=".env file committed to repository!",
                file=".env",
            ))

        return findings

    def scan(self, repo_url: str) -> ScanResult:
        """Perform full security scan of repository."""
        result = ScanResult(repo_url=repo_url)

        try:
            repo_path = self.clone_repo(repo_url)

            # Get all files
            all_files = list(repo_path.rglob("*"))
            files_to_scan = [f for f in all_files if f.is_file() and not ".git" in str(f)]
            result.files_scanned = len(files_to_scan)

            console.print(f"[dim]Scanning {len(files_to_scan)} files...[/dim]")

            # Scan each file for patterns
            for filepath in files_to_scan:
                for severity, checks in self.PATTERNS.items():
                    result.findings.extend(self.scan_file(filepath, severity, checks))

            # Check dependencies
            result.findings.extend(self.check_dependencies(repo_path))

            # Check gitignore
            result.findings.extend(self.check_gitignore(repo_path))

        finally:
            self.cleanup()

        return result


def print_results(result: ScanResult):
    """Print scan results in a nice format."""
    console.print()

    # Summary
    if result.is_safe:
        console.print(Panel(
            f"[bold green]✅ NO HIGH-SEVERITY ISSUES FOUND[/bold green]\n\n"
            f"Files scanned: {result.files_scanned}\n"
            f"Findings: {result.high_count} high, {result.medium_count} medium, {result.low_count} low",
            title="Security Scan Result",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"[bold red]🚨 SECURITY ISSUES DETECTED[/bold red]\n\n"
            f"Files scanned: {result.files_scanned}\n"
            f"Findings: [red]{result.high_count} high[/red], [yellow]{result.medium_count} medium[/yellow], {result.low_count} low",
            title="Security Scan Result",
            border_style="red",
        ))

    if not result.findings:
        console.print("\n[green]No issues found. Repository appears safe.[/green]")
        return

    # Details table
    console.print()

    # Group by severity
    for severity in ["HIGH", "MEDIUM", "LOW"]:
        findings = [f for f in result.findings if f.severity == severity]
        if not findings:
            continue

        color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "dim"}[severity]

        table = Table(
            title=f"[{color}]{severity} Severity[/{color}]",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("Category", style="cyan")
        table.add_column("Description")
        table.add_column("File", style="dim")
        table.add_column("Line", style="dim", justify="right")

        for f in findings[:20]:  # Limit to 20 per severity
            table.add_row(
                f.category,
                f.description,
                f.file.split("/repo/")[-1] if "/repo/" in f.file else f.file,
                str(f.line) if f.line else "-",
            )

        console.print(table)

        if len(findings) > 20:
            console.print(f"[dim]... and {len(findings) - 20} more {severity} findings[/dim]")

        console.print()


def check_repo(repo_url: str) -> bool:
    """
    Check a repository for security issues.

    Returns True if safe, False if issues found.
    """
    console.print(Panel(
        f"[bold]🔒 Security Scanner[/bold]\n\nTarget: {repo_url}",
        border_style="blue",
    ))

    checker = RepoSecurityChecker()

    try:
        result = checker.scan(repo_url)
        print_results(result)
        return result.is_safe
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("Usage: python security_check.py <github-repo-url>")
        console.print("Example: python security_check.py https://github.com/user/repo")
        sys.exit(1)

    repo_url = sys.argv[1]
    is_safe = check_repo(repo_url)
    sys.exit(0 if is_safe else 1)
