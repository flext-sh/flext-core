# Development Playbook

Updated to reflect the requirements of the 1.0.0 modernization plan.

---

## Tooling Stack

- **Python**: 3.13.x (local + CI)
- **Dependency Management**: Poetry (preferred) / pip
- **Formatting**: `ruff format`
- **Linting**: `ruff check`
- **Typing**: MyPy (`strict = True`)
- **Testing**: Pytest with coverage

Install everything using `make setup`.

---

## Everyday Commands

```bash
make format       # Ruff formatter
make lint         # Ruff linter
make type-check   # MyPy strict mode
make test         # Pytest (unit + integration)
make validate     # Runs format check, lint, type-check, and tests
```

`make validate` is required before opening a pull request.

---

## Contribution Guidelines

1. **Dispatcher First** – new orchestration code should interact with `FlextDispatcher`/`FlextDispatcherRegistry` rather than direct bus invocations unless a strong justification is documented.
2. **Context Awareness** – logging and telemetry must push correlation/request metadata through `FlextContext` helpers.
3. **Configuration Alignment** – bootstrap logic pulls settings from `FlextConfig` and registers them in the container.
4. **Doc Synchronisation** – README pages, docs, and docstrings must stay aligned; update them as part of every behavioural change.
5. **Tests** – add or update tests in `tests/unit`, `tests/integration`, or `tests/patterns` to cover new behaviour.

---

## Branching & Releases

- Default branch: `main`
- Release cadence: modernization milestone tagged once dispatcher/context/config pillars reach “adopted” status across pilot packages.
- Version bumps propagate through `pyproject.toml`, `src/flext_core/version.py`, and documentation snippets.

---

## IDE Tips

- Enable type checking and Ruff integration to catch issues early.
- Configure “format on save” to align with the automation pipeline.
- Use the modernization plan as the single source of truth for architectural decisions; link to plan sections in PR descriptions when relevant.

---

## Helpful Targets

```bash
make docs   # Build documentation previews (if configured)
make clean  # Remove caches (.pytest_cache, __pycache__, etc.)
```

Keep this playbook updated as tooling or modernization requirements evolve.
