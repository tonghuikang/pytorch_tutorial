For changes involving html files, please use MCP pupeeteer to test.

Jekyll Development Server
- Check if already running: lsof -i :4000 (or check /bashes for background shells)
- Start server: bundle exec jekyll serve --host 127.0.0.1 --port 4000
- View site: http://localhost:4000
- Auto-reload: Jekyll watches for file changes and rebuilds automatically
- Manual rebuild: Kill and restart server if auto-reload fails
- Test changes: Use MCP Puppeteer to navigate to http://localhost:4000

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
