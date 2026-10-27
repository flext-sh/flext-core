<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `<workspace-root>/docs/guides/using-flext-tests.md`; adjust that workspace source, never this member projection. -->

# flext-core - Using flext-tests

> Project profile: `flext-core`

<!-- TOC START -->

- [Aliases](#aliases)
- [Essential fixtures](#essential-fixtures)
- [Resetting singletons manually](#resetting-singletons-manually)
- [Testing result flows](#testing-result-flows)
- [Workspace Assertion Standard](#workspace-assertion-standard)
- [Good practices](#good-practices)
- [Make/codegen boundary](#makecodegen-boundary)
- [Bad practices](#bad-practices)
- [Related](#related)

<!-- TOC END -->

<!-- mro-wkii.17.7 (agent: codex) — keep test-toolkit guidance separate from Make/codegen ownership. -->

`flext_tests` is the shared test toolkit. It provides fixtures, matchers, file helpers,
and a test runtime that binds the canonical aliases.

## Aliases

Import the aliases each test consumes from the public `flext_tests` package root.

`flext_tests` reexports `d`, `e`, `h`, `r`, `x` from `flext_infra` and exposes domain
helpers (`tk`, `td`, `tf`, `tv`, `tm`).

| Alias | Purpose                                          |
| ----- | ------------------------------------------------ |
| `c`   | constants                                        |
| `e`   | errors / exceptions (reexported)                 |
| `m`   | models                                           |
| `p`   | protocols                                        |
| `r`   | result (reexported)                              |
| `s`   | service / test runtime (`FlextTestsServiceBase`) |
| `t`   | typings                                          |
| `u`   | utilities                                        |

**Important:** `s` is the service/test-runtime alias. Test settings are accessed via
`FlextTestsSettings` (no short alias).

## Essential fixtures

Add `flext_tests` to your project test dependencies and use these fixtures in
`conftest.py` or directly in tests:

| Fixture                  | Purpose                                                                                                                     |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `reset_settings`         | Explicit fixture resetting `FlextSettings`, `FlextTestsSettings`, and `FlextContainer` before and after its consumer.       |
| `test_runtime`           | Explicit fixture binding `c`, `e`, `m`, `p`, `r`, `t`, `u` and `service`/`settings`/`logger` on `FlextTestsCase` instances. |
| `settings`               | Clean `FlextTestsSettings(debug=True, trace=False)`.                                                                        |
| `settings_factory`       | Factory for creating project-specific settings instances.                                                                   |
| `temp_dir` / `temp_file` | Temporary paths isolated per test.                                                                                          |

With the settings plugin loaded, its `pytest_runtest_setup` and
`pytest_runtest_teardown` hooks perform automatic isolation. The two explicit fixtures
above are not declared with `autouse=True`.

```python
from __future__ import annotations

from flext_core import FlextSettings
from flext_tests import FlextTestsSettings, tm


def test_settings_isolation(settings: FlextTestsSettings) -> None:
    """The fixture settings stay distinct from the global singleton."""
    settings.debug = True
    # The settings plugin resets runtime singletons between test functions.
    tm.fail(FlextSettings.fetch_global() is settings)
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

`tm.ok(True)` validates a boolean predicate; `tm.false(False)` and
`tm.fail(False)` validate a negative boolean predicate. Boolean forms return the
unchanged boolean and accept `msg` for assertion diagnostics. Result-specific
criteria such as `path`, `has`, or `code` require a result, not a boolean.

For results, `tm.ok(result)` checks success and returns the original payload;
`tm.fail(result)` checks failure and returns the error string. Successful payloads
such as `False`, `0`, and empty collections remain successful. `r.ok(None)` is
rejected by the result constructor, not by the matcher. Neither
helper infers result state from payload truthiness. Use `tm.that(value, truth=True)`
or `truth=False` when testing general Python truthiness instead of boolean values.
`tm.that(condition)` with a boolean and no criteria requires `True`; explicit
criteria such as `tm.that(False, eq=False)` retain their declared semantics.

Assert a runtime type with `tm.that(value, is_=ExpectedType)`; it raises on a
mismatch. A boolean type probe whose result is discarded asserts nothing.

Inspect a container through `FlextTestsDocker.fetch_container_info(name)`; it is the
single owner of container status and fails `NOT_PROVISIONED` for an absent
container.

Use the `r` alias instead of importing from `returns` directly:

```python
from math import isclose

from flext_tests import p, r, tm


def safe_divide(a: float, b: float) -> p.Result[float]:
    """Divide ``a`` by ``b`` and fail explicitly on a zero divisor.

    Returns:
        The quotient, or a ``division_by_zero`` failure.

    """
    if b == 0:
        return r[float].fail("division_by_zero")
    return r[float].ok(a / b)


def test_safe_divide() -> None:
    """Division succeeds for a nonzero divisor and fails for zero."""
    quotient = tm.ok(safe_divide(10, 2))
    tm.ok(isclose(quotient, 5.0))
    tm.fail(safe_divide(10, 0))
```

## Workspace Assertion Standard

This contract applies to first-party workspace roots, attached members, and
standalone packages. `flext-tests` owns matcher behavior; the universal
`rewire-test-assert-*` catalog in `flext-infra` owns automated migration. Project
facades may reexport the canonical `tm`, but never implement a competing matcher
or a project-specific assertion rewrite.

| Intent | Public pattern | Required invariant |
| --- | --- | --- |
| Successful or failed Result | `tm.ok(result)` / `tm.fail(result)` | State is independent of payload truthiness; success returns the original payload. |
| Native equality or inequality | `tm.that(value, eq=expected)` / `ne=expected` | Operators are independent; operands stay native and successful checks do not render them. |
| General truth | `tm.that(value, truth=True)` / `truth=False` | The supplied subject is observed without payload normalization. |
| Typed local value | `value = tm.not_none(value)` / `tm.instance(value, Expected)` | Return the same object; rebind only when that local binding is intended. |
| Guard, short-circuit condition, or lazy message | Native conditional branch with `tm.reject()` / `tm.reject(message)` | Preserve conditional evaluation, narrowing, name scope, exception arguments, and message timing. |
| Structured input | Validate once with the canonical Pydantic model, then assert its public behavior | Do not scatter casts, dictionary narrowing, or schema copies through consumers. |
| Configured expectations | Read the same typed config/settings owner as production | A valid configuration change must not invalidate a hardcoded expectation. |

The automation retains native control flow when a matcher call alone cannot prove
the original semantics. It does not rebind globals, reinterpret a callable message
as a factory, or replace assertion failures with a different exception category.
Handwritten matcher criteria remain available when their declared behavior is the
intended contract.

Apply the same catalog per discovered repository, not a copied repository list.
Admit the actual `tm` binding, publish authenticated file plans, and rescan the
affected scope to a fixed point. Preserve valid independently published effects;
refresh changed inputs without discarding shared work or weakening admission.
Generated test skeletons and project guidance come from the `flext-infra`
templates, not handwritten projections in each member.

Each repository needs its own runtime, test, lint, and type evidence using its
unchanged `pyproject.toml`. A green matcher package or one migrated member is not
workspace completion. Unexecuted external capabilities remain unexecuted, and
unresolved or unadmitted findings remain open rather than being classified green.

## Good practices

- Rely on `reset_settings` and `test_runtime` for isolation.
- Assert public API behavior, not private internals.
- `tm.that(value, eq=expected)` and `ne=expected` use Python's equality and
  inequality operators independently. Passing checks do not render the subject;
  comparison and predicate exceptions escape before diagnostic formatting.
- Use `tm.not_none(value)` or `tm.instance(value, Expected)` when the returned
  value must retain narrowing. For compound guards, call `tm.reject()` only in
  the failing branch; it returns `Never` and raises native `AssertionError`.
  Pass a message inside that branch to preserve lazy evaluation and its original
  object identity instead of eagerly formatting it.
- Matcher arguments are ordinary Python arguments: a matcher cannot recover
  short-circuit evaluation or narrowing already lost before its call. Automated
  conversion therefore retains complex conditions in native control flow and
  uses `tm.reject()` for the failing branch. It must not introduce assignments
  that change the election of global, nonlocal, or local names.
- Use `settings_factory` when a project-specific settings subclass is required.
- Use `tm.ok(result)` and `tm.fail(result)` for result assertions; inspect state
  properties directly only when that state query is the behavior under test.

## Make/codegen boundary

`flext_tests` owns test fixtures, models, assertions, and public-behavior test support
only. It does not own a Make registry, dispatcher, generator, or workspace inventory.

Repository conformance and the complete generated Makefile are owned solely by
`flext-infra codegen conform`. Discover the current selector-free verbs through
`make help` in the owning repository root. Each verb executes its operation directly; do
not add a `WHAT` selector or duplicate the dispatcher in a test helper.

Tests for this contract exercise the generated public commands and observable artifacts.
They do not reproduce command metadata or assert private routing implementation. See
ADR-004 for the
canonical decision.

## Bad practices

Do not mutate a global singleton without the settings plugin or explicit reset fixture
providing isolation. Do not import a separate result implementation from `returns`;
exercise the public `r` facade used by the production consumer.

For a standalone test without the settings plugin, keep the reset on both sides of the
mutation, including assertion failure:

```python
from flext_core import FlextSettings
from flext_tests import tm


def test_settings_override() -> None:
    """A debug override is visible and reset on both sides of the test."""
    FlextSettings.reset_for_testing()
    try:
        settings = FlextSettings.fetch_global()
        settings.debug = True
        tm.ok(settings.debug)
    finally:
        FlextSettings.reset_for_testing()
```

## Related

- `flext-tests/src/flext_tests/_fixtures/settings.py`
- Architecture ADR 004
- Testing standards
