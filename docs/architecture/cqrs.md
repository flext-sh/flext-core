# CQRS Architecture

<!-- TOC START -->

- [Overview](#overview)
- [Executable Dispatcher Flow](#executable-dispatcher-flow)
- [Command Handling with r\[T\]](#command-handling-with-rt)
- [Context and Metrics (Illustrative Internals)](#context-and-metrics-illustrative-internals)
- [Best Practices](#best-practices)

<!-- TOC END -->

## Overview

FLEXT CQRS orchestration centers on dispatcher + handlers + result contracts. This page
uses executable examples and text-only internals where appropriate.

## Executable Dispatcher Flow

```python
from examples.ex_04_flext_dispatcher import Ex04DispatchDsl

result = Ex04DispatchDsl.run()
expected_value = "pong:dispatcher-example"
if not result.success:
    message = "Expected dispatcher success"
    raise RuntimeError(message)
if result.value != expected_value:
    message = "Unexpected dispatcher value"
    raise RuntimeError(message)
```

## Command Handling with r[T]

```python
from __future__ import annotations

from flext_core import p, r, t


def create_user(command: t.StrDict) -> p.Result[t.StrDict]:
    """Create a user payload when the required email is present.

    Returns:
        The resulting ``p.Result[t.StrDict]``.

    """
    if "email" not in command:
        return r[t.StrDict].fail("missing_email")
    return r[t.StrDict].ok({"status": "created", "email": command["email"]})


ok = create_user({"email": "user@example.com"})
ko = create_user({})

if not ok.success:
    message = "Expected created user success"
    raise RuntimeError(message)
if not ko.failure:
    message = "Expected missing-email failure"
    raise RuntimeError(message)
```

## Context and Metrics (Illustrative Internals)

```text
Internal state sketch (illustrative):

self._context_stack: t.SequenceOf[t.JsonMapping] = []
self._metrics = {}

Typical runtime interactions:
handler.push_context({c.ContextScope.OPERATION: "create_user"})
handler.record_metric("users_created", 1)
metrics = handler.get_metrics()
handler.pop_context()
```

## Best Practices

- Keep handlers small and deterministic.
- Return `p.Result[T]` for all fallible command/query paths.
- Use dispatcher orchestration from established examples before adding custom layers.
