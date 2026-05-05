# Service Patterns

<!-- TOC START -->
- [Overview](#overview)
- [Simple Service Pattern](#simple-service-pattern)
- [Validate Before Execute](#validate-before-execute)
- [examples-backed service flows](#examples-backed-service-flows)
<!-- TOC END -->

## Overview

Service classes should keep orchestration explicit and failures modeled with `r[T]`.

## Simple Service Pattern

```python
from flext_core import p, r, s


class CreateUserService(s):
    def execute(self) -> p.Result[str]:
        return r[str].ok("user_created")


service = CreateUserService()
result = service.execute()
assert result.success
assert result.value == "user_created"
```

## Validate Before Execute

```python
from typing import Annotated

from flext_core import m, p, r, s


class ValidateThenCreateService(s):
    username: Annotated[str, m.Field(description="Username for the create flow.")] = ""

    def execute(self) -> p.Result[str]:
        if not self.username:
            return r[str].fail("username_required")
        return r[str].ok(self.username)


assert ValidateThenCreateService(username="alice").execute().success
assert ValidateThenCreateService(username="").execute().failure
```

## examples-backed service flows

```python
from examples.ex_11_flext_service import ExampleService

ExampleService.run()
```
