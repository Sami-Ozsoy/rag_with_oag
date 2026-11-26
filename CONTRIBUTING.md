# Contributing

## Development setup

Use Python 3.12 or newer. On macOS, install OpenMP for embedded FalkorDB with
`brew install libomp`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the CI checks from the repository root.

```bash
ruff check .
ruff format --check .
mypy
pytest --cov=rag_with_oag --cov-report=term-missing
```

The Ruff format check covers the whole repository, including Python code blocks in Markdown.
Tests use embedded FalkorDB and fake models and run without network or Docker.

See [Evaluation](docs/evaluation.md) for retrieval and answer accuracy runs.
Run `python spikes/m0_falkordb.py` to measure ANN recall, hybrid latency and
memory in the FalkorDB spike.

## Pull requests

Keep pull requests small and focused. Add tests for behaviour changes and
update the relevant docs and [CHANGELOG.md](CHANGELOG.md).

Report security issues as described in [SECURITY.md](SECURITY.md).

## Releasing

Bump the version in `pyproject.toml` and update [CHANGELOG.md](CHANGELOG.md), then run `git tag vX.Y.Z && git push origin vX.Y.Z`. Pre-release tags such as `v0.3.0rc1` go only to TestPyPI. Publishing uses Trusted Publishing through the `pypi` GitHub environment without API tokens.
