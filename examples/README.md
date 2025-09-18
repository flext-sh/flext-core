# FLEXT Core Examples

Curated examples that demonstrate the patterns guaranteed for the 1.0.0 release: unified dispatcher flows, context-first observability, and aligned configuration/domain services.

---

## Example Map

| File                                 | Focus                                        | 1.0.0 Pillar                |
| ------------------------------------ | -------------------------------------------- | --------------------------- |
| `01_railway_result.py`               | FlextResult basics                           | Railway foundation          |
| `02_dependency_injection.py`         | Container registration/resolution            | Configuration & services    |
| `03_cqrs_commands.py`                | Command/query handlers using `FlextBus`      | Unified dispatcher          |
| `04_validation_modern.py`            | Validation helpers built on FlextResult      | Railway foundation          |
| `05_validation_advanced.py`          | Complex validation flows                     | Railway foundation          |
| `06_ddd_entities_value_objects.py`   | Entities, values, aggregates                 | Configuration & services    |
| `07_mixins_multiple_inheritance.py`  | Mixins for behaviour composition             | Configuration & services    |
| `08_configuration.py`                | Layered `FlextConfig` patterns               | Configuration & services    |
| `10_events_messaging.py`             | Event-driven orchestration                   | Unified dispatcher          |
| `11_handlers_pipeline.py`            | Handler pipeline composition                 | Unified dispatcher          |
| `12_logging_structured.py`           | `FlextLogger` + `FlextContext` integration   | Context-first observability |
| `13_architecture_interfaces.py`      | Protocol-driven boundaries                   | Configuration & services    |
| `14_exceptions_handling.py`          | Structured exceptions + results              | Railway foundation          |
| `15_advanced_patterns.py`            | Multi-step workflows with dispatcher         | Unified dispatcher          |
| `16_integration.py`                  | Integrating container + config + dispatcher  | All pillars                 |
| `17_end_to_end.py`                   | Full workflow (config → dispatcher → domain) | All pillars                 |
| `18_semantic_modeling.py`            | Semantic and value modelling                 | Configuration & services    |
| `19_modern_showcase.py`              | Modern application bootstrap                 | All pillars                 |
| `20_boilerplate_reduction.py`        | Helpers that remove repetitive setup         | Configuration & services    |
| `21_basic_result_patterns.py`        | Lightweight FlextResult recipes              | Railway foundation          |
| `22_monadic_composition_advanced.py` | Advanced map/flat_map usage                  | Railway foundation          |
| `23_test_utilities_demo.py`          | Using flext-tests factories                  | Tooling support             |

Shared utilities that underpin the examples live in `shared_example_strategies.py`.

---

## Running Examples

```bash
cd flext-core
poetry install  # or make setup

python examples/01_railway_result.py
python examples/12_logging_structured.py
python examples/17_end_to_end.py
```

The scripts rely on the local checkout (no network access required). Each example prints contextual output that illustrates how `FlextContext` metadata flows through handlers.

---

## Modernization Checklist

All new examples must:

1. Use `FlextDispatcher` or `FlextBus` via the dispatcher façade when demonstrating command flows.
2. Populate `FlextContext` metadata (correlation ID, operation name) when logging.
3. Load configuration through `FlextConfig` subclasses or container bootstrap utilities.

Follow this checklist when contributing additional examples ahead of the 1.0.0 release.
