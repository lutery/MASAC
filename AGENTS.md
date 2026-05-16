# AGENTS.md

## Build & Run

- **Package manager**: [Poetry](https://python-poetry.org/). Use `poetry install`, `poetry run <cmd>`.
- **Python**: ≥3.9, <3.12 (see `pyproject.toml`). The `<3.12` constraint is because `distutils.util.strtobool` is removed in 3.12.

```shell
poetry install
poetry run python masac/masac.py        # Torch implementation
poetry run python masac/masac_jax.py    # JAX implementation
```

## Lint & Format (pre-commit)

```shell
pre-commit run --all-files
```

Rules: **black** (line-length=127), **isort** (profile=black, `src_paths=["masac"]`), **flake8** (`--max-line-length=456`, ignores E203, W503, E741), **pyupgrade** (`--py37-plus`), **codespell**. No typecheck (no mypy configured).

## Tests

There is no test suite. Pytest is listed in dependencies but no test files exist.

## Architecture

Single package `masac/` with two independent entrypoints (both are `if __name__ == "__main__"` scripts):

| File | Framework | Model save |
|------|-----------|------------|
| `masac/masac.py` | PyTorch (`torch`) | `actor.pth` (root) |
| `masac/masac_jax.py` | JAX + Flax + Optax | `trained_model/actor/` (orbax) |

Shared modules:
- `masac/ma_buffer.py` — `MAReplayBuffer`, `Experience` namedtuple
- `masac/utils.py` — `extract_agent_id(agent_str)`

**Critical: import style matters.** Within the `masac/` package, scripts import from their own directory (e.g. `from ma_buffer import ...`), not the package (`from masac.ma_buffer import ...`). The latter only appears in `__init__.py`. If you refactor or add tests outside the package, use `from masac.ma_buffer import ...`.

## Algorithm-Specific Constraints

- **Homogeneous agents only**: All agents share the same action and observation spaces.
- **Agent ID format**: IDs must be strings like `"agent_0"`, `"agent_1"`, etc. (parsed by `extract_agent_id`). The code assumes agent IDs are contiguous integers 0..n-1.
- **Shared parameters**: One critic (conditioned on global state + joint actions), one actor (conditioned on local obs + agent ID).
- **Default env**: `simple_spread_v3` (PettingZoo MPE) with 3 agents, continuous actions.
- **Action normalization differs**: The JAX implementation normalizes stored actions to [-1, 1] and denormalizes on sampling. The Torch implementation does not.

## CI

Only pre-commit linting runs on PR and push to `main`. No test or build workflow. See `.github/workflows/pre-commit.yaml`.
