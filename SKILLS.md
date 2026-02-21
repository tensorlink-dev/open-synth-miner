# Agent Skills & Integrations

Tools, MCP servers, AI agents, and CI bots that can augment development on Open Synth Miner.

---

## MCP Servers

MCP (Model Context Protocol) servers extend AI coding agents with structured tool access. Add these to `.mcp.json` at the repo root or configure per-agent.

### Weights & Biases

Query runs, metrics, sweeps, and Weave traces. Create reports programmatically.

```json
{
  "mcpServers": {
    "wandb": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/wandb/wandb-mcp-server", "wandb_mcp_server"],
      "env": { "WANDB_API_KEY": "<your-key>" }
    }
  }
}
```

**Tools:** `query_wandb_tool` (GraphQL queries for runs/sweeps), `create_wandb_report_tool` (programmatic reports), `query_wandb_entity_projects`, `query_weave_traces_tool`, `count_weave_traces_tool`, `query_wandb_support_bot`

### Hugging Face Hub

Search models, datasets, papers, and Spaces on the Hub. Remote server — no local process needed.

```json
{
  "mcpServers": {
    "huggingface": {
      "type": "http",
      "url": "https://huggingface.co/mcp",
      "headers": { "Authorization": "Bearer <your-hf-token>" }
    }
  }
}
```

**Tools:** model search/details, dataset search/details, Space search, paper search. Configure exposed tools at [huggingface.co/settings/mcp](https://huggingface.co/settings/mcp).

### GitHub

Browse repos, manage issues/PRs, trigger Actions, search code.

```json
{
  "mcpServers": {
    "github": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-pat>" }
    }
  }
}
```

**Tools:** repos, issues, pull_requests, code_security, actions, users, context

### arXiv

Search, download, and read ML papers. Useful for researching new architectures or referencing prior work.

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "uvx",
      "args": ["arxiv-mcp-server", "--storage-path", "./papers"]
    }
  }
}
```

Install: `uv tool install arxiv-mcp-server`

**Tools:** search papers (with category filters like `cs.LG`, `q-fin.CP`), download by arXiv ID, list/read downloaded papers

### Combined `.mcp.json`

Drop this at the repo root for the full stack:

```json
{
  "mcpServers": {
    "wandb": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/wandb/wandb-mcp-server", "wandb_mcp_server"],
      "env": { "WANDB_API_KEY": "<your-key>" }
    },
    "huggingface": {
      "type": "http",
      "url": "https://huggingface.co/mcp",
      "headers": { "Authorization": "Bearer <your-hf-token>" }
    },
    "github": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-pat>" }
    },
    "arxiv": {
      "command": "uvx",
      "args": ["arxiv-mcp-server", "--storage-path", "./papers"]
    }
  }
}
```

---

## AI Coding Agents

### GitHub Copilot Coding Agent
Assign GitHub issues directly to Copilot. It spins up a sandbox, reads the codebase, writes code, runs tests, and opens a draft PR. Available on paid Copilot plans — enable under repo Copilot policies.

Best for: well-scoped issues like "add unit test for X" or "refactor config loading."

### CodeRabbit
AI-powered PR reviewer. Installs as a GitHub App from the [Marketplace](https://github.com/marketplace/coderabbitai). Auto-posts line-by-line reviews, summaries, and catches bugs on every PR. Interact via `@coderabbitai` comments. Free for open-source repos.

Best for: catching tensor shape mismatches, unused imports, logic errors in scoring functions.

### Devin
Autonomous AI engineer by Cognition Labs. Assign tasks via natural language — it plans, codes, tests, and opens PRs in a sandboxed cloud environment. $20/month core plan. Connect your GitHub repo at [app.devin.ai](https://devin.ai).

Best for: well-defined tasks with clear acceptance criteria. Always review output carefully.

---

## CI/CD Bots

### GitHub Actions (pytest)

Runs the test suite on every push and PR. Create `.github/workflows/test.yml`:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}
      - run: pip install -e .
      - run: pip install pytest
      - run: pytest --tb=short -q
```

### pre-commit.ci + Ruff

Auto-formats and lints on every PR. Install the [pre-commit.ci GitHub App](https://github.com/apps/pre-commit-ci) and add `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.0
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=5000']
```

`check-added-large-files` is critical for ML repos to prevent accidental model weight commits. Free for open-source.

### Dependabot

Auto-opens PRs for dependency updates and security patches. Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: pip
    directory: "/"
    schedule:
      interval: weekly
  - package-ecosystem: github-actions
    directory: "/"
    schedule:
      interval: weekly
```

### W&B GitHub Action

Surfaces experiment metrics in PRs. Correlates W&B runs with the commit that produced them.

```yaml
- uses: wandb/wandb-action@v1
  with:
    WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
    FILTER_GITHUB_SHA: ${{ github.sha }}
```

### HuggingFace Hub Push

Auto-publishes model checkpoints to the Hub on merge:

```yaml
- uses: backendcloud/hugging-push@v0.2.3
  with:
    huggingface_token: ${{ secrets.HF_TOKEN }}
    huggingface_repo: tensorlink-dev/SN50-Hybrid-Hub
    repo_type: model
```

---

## Recommended Setup Priority

| # | Tool | Effort | What it does |
|---|------|--------|--------------|
| 1 | GitHub Actions + pytest | ~30 min | Catches regressions on every PR |
| 2 | pre-commit.ci + Ruff | ~15 min | Auto-format, lint, block large files |
| 3 | Dependabot | ~5 min | Automated dependency security updates |
| 4 | CodeRabbit | ~2 min | Free AI code review on every PR |
| 5 | W&B MCP server | ~10 min | Query experiments from your agent |
| 6 | HuggingFace MCP server | ~5 min | Search models/datasets from your agent |
| 7 | W&B GitHub Action | ~20 min | Experiment metrics in PRs |
| 8 | HF Hub push action | ~15 min | Auto-publish models on merge |
| 9 | arXiv MCP server | ~5 min | Research papers from your agent |
| 10 | Copilot Coding Agent | ~5 min | Delegate backlog issues to AI |
