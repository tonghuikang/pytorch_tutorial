"""
Claude Code Hook: Post-Bash Command Validator.

Validates bash commands after execution.
"""


def validate_bash_command(command: str) -> list[str]:
    """Validate bash command after execution."""
    issues = []

    if " && " in command:
        issues.append("Please try to run the commands individually.")

    if "rg" in command and "--no-ignore" not in command:
        issues.append("If you intend to search .venv folder, please use rg --no-ignore.")

    return issues
