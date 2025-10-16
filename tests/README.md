# FLEXT Core Test Suite

Test coverage aligns with the 1.0.0 modernization pillars: dispatcher unification, context-first observability, and shared configuration/runtime services.

---

## Layout

```
tests/
├── unit/                # Module-level guarantees (result, container, dispatcher, etc.)
├── integration/         # Cross-module flows (config + container + dispatcher)
├── patterns/            # Behavioural contracts shared with ecosystem packages
└── conftest.py          # Fixtures that bootstrap FlextContext and the global container
```

Notable files:

- `unit/test_dispatcher.py` – validates handler registration and metadata propagation.
- `unit/test_context.py` – enforces context correlation semantics.
- `unit/test_container_100_percent.py` – ensures DI guarantees ahead of ecosystem adoption.
- `integration/test_wildcard_exports_clean.py` – protects the public API surface.
- `patterns/test_patterns_commands.py` – codifies CQRS usage expected downstream.

---

## Running the Suite

```bash
poetry run pytest tests/unit -q
poetry run pytest tests/integration -m "not slow"
poetry run pytest --cov=src/flext_core --cov-report=term-missing
```

For dispatcher pilots, run the focused marker:

```bash
poetry run pytest tests/patterns -m dispatcher
```

---

## Modernization Expectations

- All new tests must exercise `FlextDispatcher` + `FlextContext` interactions where applicable.
- Coverage target: keep the baseline ≥79% while pushing towards the 85% goal called out in the modernization plan.
- Fixtures should rely on `FlextConfig` and `FlextContainer` bootstrap helpers rather than bespoke setup code.

Please update this README if new suites or markers are introduced as part of the 1.0.0 workstream.
