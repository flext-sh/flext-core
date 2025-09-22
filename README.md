# FLEXT-Core

Foundation library for the FLEXT ecosystem that provides railway-oriented programming, dependency injection, and domain modelling with Python 3.13+ type safety.

> **Status**: v0.9.9 Release Candidate powering the 1.0.0 modernization programme. · 1.0.0 Release Preparation

---

## Modernization Pillars for 1.0.0

The FLEXT Core Modernization Plan identifies three pillars that must land for the 1.0.0 release. Every README, doc page, and docstring in this repository now reflects these commitments:

1. **Unified Dispatcher Flow** – `FlextDispatcher` and `FlextDispatcherRegistry` standardise command/query routing across CLI and connector packages while preserving existing `FlextBus` semantics.
2. **Context-First Observability** – `FlextContext` becomes the single entry point for correlation IDs, request metadata, and latency tracking. Legacy ad-hoc context helpers are considered migration candidates.
3. **Configuration & Domain Service Alignment** – `FlextConfig`, `FlextContainer`, and `FlextDomainService` define the default runtime contract so every project shares the same configuration lifecycle and domain-service ergonomics.

---

## What Ships Today

- `FlextResult[T]` railway pattern with dual `.value`/`.data` access maintained for ABI stability.
- Singleton `FlextContainer` with typed service keys, container reset utilities, and dispatcher integration helpers.
- Domain-driven patterns through `FlextModels` and `FlextDomainService` with context-aware logging.
- Structured logging (`FlextLogger`) wired to `FlextContext` for correlation and latency metrics.
- Layered configuration loader (`FlextConfig`) with `.env`, TOML, and YAML support, now enforcing canonical environment and log
  level values via literal types.
- `FlextDispatcher` façade and registry to prepare CLI and connector packages for unified dispatch.

---

## Quick Start

```bash
git clone https://github.com/flext-sh/flext-core.git
cd flext-core
make setup
python -c "from flext_core import FlextResult; print('FLEXT-Core ready for 1.0.0')"
```

```python
from flext_core import FlextContainer, FlextDispatcher, FlextLogger, FlextResult

container = FlextContainer.get_global()
container.register("logger", FlextLogger("example"))

dispatcher = FlextDispatcher()

class PingHandler:
    def handle(self, command: dict[str, str]) -> FlextResult[str]:
        container.get("logger").unwrap().info("ping", extra=command)
        return FlextResult[str].ok("pong")

dispatcher.register_command(dict, PingHandler())
response = dispatcher.dispatch({"message": "ping"})
print(response.unwrap())
```

---

## Module Guide

| Area                        | Modules                                                                             | Release Focus                      |
| --------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------- |
| **Foundation**              | `result.py`, `typings.py`, `constants.py`                                           | Maintain ABI and type guarantees   |
| **Runtime Surfaces**        | `container.py`, `domain_services.py`, `models.py`, `utilities.py`                   | Shared service and domain patterns |
| **Execution Flow**          | `bus.py`, `dispatcher.py`, `dispatcher_registry.py`, `handlers.py`, `processing.py` | Unified dispatcher roadmap         |
| **Context & Observability** | `context.py`, `loggings.py`                                                         | Context-first adoption and metrics |
| **Configuration**           | `config.py`, `version.py`                                                           | Single configuration lifecycle     |

See `docs/architecture/flext_modernization_plan.md` for the detailed modernization charter.

---

## Development Workflow

```bash
make format
make lint
make type-check
make test
```

- Coverage baseline: 79% (target ≥85% for 1.0.0).
- Supported interpreter: Python 3.13.x (validated in CI).
- Linting: Ruff; typing: MyPy strict mode.

---

## Test Topology

```
tests/
├── unit/                # Coverage-oriented leaf tests
├── integration/         # Dispatcher, container, config lifecycle
├── patterns/            # CQRS and domain pattern contract checks
└── conftest.py          # Shared fixtures and context bootstrap
```

Typical commands:

```bash
pytest tests/unit/test_result_100_percent_coverage.py
pytest tests/integration/test_wildcard_exports_clean.py
pytest tests/patterns -m dispatcher
```

---

## Roadmap Snapshot (Modernization Plan)

- **Phase 0 – Baseline** (complete): inventory of dispatchers, handlers, and configuration entry points across the ecosystem.
- **Phase 1 – Dispatcher Charter** (in-flight): `FlextDispatcher` façade published, registry helpers documented, CLI/Oracle pilot migrations identified.
- **Phase 2 – Tooling & Docs** (week 4–5): finalize documentation in this repository, ship lint rules enforcing dispatcher usage, publish migration playbooks.
- **Phase 3 – Ecosystem Migration**: downstream packages adopt dispatcher/context shims with playbook support and regression coverage.
- **Phase 4 – Deprecations**: flag bespoke bus/dispatcher helpers for removal once adoption thresholds are met.

All documentation stays synchronized with `docs/architecture/flext_modernization_plan.md`.

---

## Contributions

- Pass `make validate` before opening a PR.
- Prefer `FlextDispatcher` integration for new command-flow examples.
- Surface context metadata via `FlextContext` helpers when introducing logging.
- Keep docs and docstrings in sync with behavioural changes.

---

## Support

- Documentation: `flext-core/docs/`
- Issues: GitHub tracker (`flext-sh/flext-core`)
- Security: private disclosure to the FLEXT maintainers

---

FLEXT-Core remains the stable foundation for 32+ FLEXT packages while the 1.0.0 modernization plan unifies dispatcher usage, context propagation, and configuration lifecycles across the ecosystem.
