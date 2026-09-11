<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `docs/guides/using-flext-tests.md`; adjust that source, never this projection. -->

# flext-core - Using flext-tests

> Project profile: `flext-core`

<!-- TOC START -->
- [Aliases](#aliases)
- [Essential fixtures](#essential-fixtures)
- [Resetting singletons manually](#resetting-singletons-manually)
- [Testing result flows](#testing-result-flows)
- [Good practices](#good-practices)
- [Make/codegen boundary](#makecodegen-boundary)
- [Bad practices](#bad-practices)
- [Related](#related)
<!-- TOC END -->

<!-- mro-wkii.17.7 (agent: codex) — keep test-toolkit guidance separate from Make/codegen ownership. -->

`flext_tests` is the shared test toolkit. It provides fixtures, matchers, file helpers, and a test runtime that binds
the canonical aliases.

## Aliases

Import the aliases each test consumes from the public `flext_tests` package root.

`flext_tests` reexports `d`, `e`, `h`, `r`, `x` from `flext_infra` and exposes domain helpers (`tk`, `td`, `tf`, `tv`,
`tm`).

| Alias | Purpose |
| ------- | --------- |
| `c` | constants |
| `e` | errors / exceptions (reexported) |
| `m` | models |
| `p` | protocols |
| `r` | result (reexported) |
| `s` | service / test runtime (`FlextTestsServiceBase`) |
| `t` | typings |
| `u` | utilities |

**Important:** `s` is the service/test-runtime alias. Test settings are accessed via `FlextTestsSettings` (no short
alias).

## Essential fixtures

Add `flext_tests` to your project test dependencies and use these fixtures in `conftest.py` or directly in tests:

| Fixture | Purpose |
| --------- | --------- |
| `reset_settings` | Explicit fixture resetting `FlextSettings`, `FlextTestsSettings`, and `FlextContainer` before and after its consumer. |
| `test_runtime` | Explicit fixture binding `c`, `e`, `m`, `p`, `r`, `t`, `u` and `service`/`settings`/`logger` on `FlextTestsCase` instances. |
| `settings` | Clean `FlextTestsSettings(debug=True, trace=False)`. |
| `settings_factory` | Factory for creating project-specific settings instances. |
| `temp_dir` / `temp_file` | Temporary paths isolated per test. |

With the settings plugin loaded, its `pytest_runtest_setup` and
`pytest_runtest_teardown` hooks perform automatic isolation. The two explicit
fixtures above are not declared with `autouse=True`.

```python
from __future__ import annotations

from flext_core import FlextSettings
from flext_tests import FlextTestsSettings


def test_settings_isolation(settings: FlextTestsSettings) -> None:
    settings.debug = True
    # The settings plugin resets runtime singletons between test functions.
    assert FlextSettings.fetch_global() is not settings
```

## Resetting singletons manually

When a fixture is not enough:

```python
from flext_core import FlextContainer, FlextSettings
from flext_tests import FlextTestsSettings

FlextSettings.reset_for_testing()
FlextTestsSettings.reset_for_testing()
FlextContainer.reset_for_testing()
```

## Testing result flows

Use the `r` alias instead of importing from `returns` directly:

```python
from flext_tests import r


def safe_divide(a: float, b: float) -> r[float]:
    if b == 0:
        return r[float].fail("division_by_zero")
    return r[float].ok(a / b)


def test_safe_divide() -> None:
    result = safe_divide(10, 2)
    assert result.success
    assert result.unwrap() == 5.0

    failure = safe_divide(10, 0)
    assert failure.failure
```

## Good practices

- Rely on `reset_settings` and `test_runtime` for isolation.
- Assert public API behavior, not private internals.
- Use `settings_factory` when a project-specific settings subclass is required.
- Assert result state via `.success`, `.failure`, and `.unwrap()` on `r[T]` instances.

## Make/codegen boundary

`flext_tests` owns test fixtures, models, assertions, and public-behavior test
support only. It does not own a Make registry, dispatcher, generator, or
workspace inventory.

Repository conformance and the complete generated Makefile are owned solely by
`flext-infra codegen conform`. Discover the current selector-free verbs through
`make help` in the owning repository root. Mutating execution requires `APPLY=Y`;
do not add a `WHAT` selector or duplicate the dispatcher in a test helper.

Tests for this contract exercise the generated public commands and observable
artifacts. They do not reproduce command metadata or assert private routing
implementation. See
ADR-004 for
the canonical decision.

## Bad practices

Do not mutate a global singleton without the settings plugin or explicit reset
fixture providing isolation. Do not import a separate result implementation from
`returns`; exercise the public `r` facade used by the production consumer.

For a standalone test without the settings plugin, keep the reset on both sides
of the mutation, including assertion failure:

```python
from flext_core import FlextSettings


def test_settings_override() -> None:
    FlextSettings.reset_for_testing()
    try:
        settings = FlextSettings.fetch_global()
        settings.debug = True
        assert settings.debug
    finally:
        FlextSettings.reset_for_testing()
```

## Related

- `.agents/skills/using-flext-tests/SKILL.md`
- `.agents/skills/coding-standards/SKILL.md`
- `flext-tests/src/flext_tests/_fixtures/settings.py`
- `docs/architecture/adr/004-generic-make-framework-in-flext-tests.md`
