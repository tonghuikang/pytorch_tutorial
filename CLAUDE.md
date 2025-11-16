For changes involving html files, please use MCP pupeeteer to test.

Package Management
- ONLY use uv, NEVER pip
- Installation: uv add package
- Running tools: uv run tool
- Upgrading: uv add --dev package --upgrade-package package
- FORBIDDEN: uv pip install, @latest syntax

Formatting
- Format: uv run --frozen ruff format *.py
- Check: uv run --frozen ruff check *.py
- Fix: uv run --frozen ruff check *.py --fix
- Sort imports: uv run --frozen ruff check --select I *.py --fix
- Type checking: uv run --frozen mypy *.py

Note
- When searching files, please always use Bash(rg)
    - You might need to use `rg --no-ignore` to search files in `.venv`
    - You might be reminded to use --no-ignore
        - But this reminder may not always be relevant.
