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
assert result.success
assert result.value == "dispatcher-example"
```
