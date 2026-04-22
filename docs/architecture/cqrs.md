# CQRS Architecture

## Overview

FLEXT CQRS orchestration centers on dispatcher + handlers + result contracts.
This page uses executable examples and text-only internals where appropriate.

## Executable Dispatcher Flow

```python
from examples.ex_04_flext_dispatcher import Ex04DispatchDsl

result = Ex04DispatchDsl.run()
assert result.success
assert result.value == "dispatcher-example"
```

## Command Handling with r[T]

```python
from flext_core import p, r


def create_user(command: dict[str, str]) -> p.Result[dict[str, str]]:
    if "email" not in command:
        return r[dict[str, str]].fail("missing_email")
    return r[dict[str, str]].ok({"status": "created", "email": command["email"]})


ok = create_user({"email": "user@example.com"})
ko = create_user({})

assert ok.success
assert ko.failure
```

## Context and Metrics (Illustrative Internals)

```text
Internal state sketch (illustrative):

self._context_stack: Sequence[Mapping[str, t.Container]] = []
self._metrics: Mapping[str, t.Container] = {}

Typical runtime interactions:
handler.push_context({"operation": "create_user"})
handler.record_metric("users_created", 1)
metrics = handler.get_metrics()
handler.pop_context()
```

## Best Practices

- Keep handlers small and deterministic.
- Return `p.Result[T]` for all fallible command/query paths.
- Use dispatcher orchestration from established examples before adding custom layers.
