<!-- TOC START -->

- [Core Layers](#core-layers)
- [Executable CQRS reference](#executable-cqrs-reference)

<!-- TOC END -->

# Architecture Overview

## Core Layers

- L3: Application orchestration
- L2: Domain behaviors
- L1: Runtime/foundation utilities
- L0: Contracts and typing boundaries

## Executable CQRS reference

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
