"""
Claude Code Hook: Pre-Bash Command Validator.

Validates bash commands before execution.
"""


def validate_before_execution(command: str) -> list[str]:
    """Validate bash command before execution."""
    issues = []

    if command.startswith("python"):
        issues.append("Please use `uv run python ...`")

    if command.startswith("grep"):
        issues.append("Please use the Bash(rg) instead grep.")

    if command.startswith("find"):
        issues.append("Please use the Bash(rg) instead of find.")

    return issues
