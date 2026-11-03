# Pydantic v2 Patterns

<!-- TOC START -->

- [Overview](#overview)
- [Native Decorators](#native-decorators)
- [Base Model + Field](#base-model-field)
- [ConfigDict + model\_dump](#configdict-model_dump)
- [field\_validator](#field_validator)
- [examples-backed sanity check](#examples-backed-sanity-check)

<!-- TOC END -->

## Overview

This guide shows practical Pydantic v2 patterns used in `flext-core`.

## Native Decorators

Use `m.field_serializer`, `m.model_serializer`, and `m.computed_field` through
the public facade. The corresponding `u` declarations inherit the same owner
through MRO; they do not implement another serializer or computed-field path.

The model owner binds the native Pydantic callables as typed `ClassVar` factories.
Their protocols preserve the bare and configured overloads, plain and wrap modes,
callback bounds, and keyword contracts. Wrapping an overloaded factory in another
`staticmethod` constructor can discard overloads during type inference and is not
the declaration pattern used here. `t.ComputedFieldInfo` provides the native
metadata annotation for computed-field title callbacks.

Validate these declarations by constructing real models and checking public
validation, property access, Python dumps, and JSON dumps. Wrap-serializer tests
exercise the real native handler; tests must not replace it or accept an unchanged
payload in place of the declared transformation.

An internal field can expose a structural `p` contract while retaining its canonical
model's validation and serialization schema: annotate it with
`Annotated[p.Contract, m.GetPydanticSchema(lambda _source, handler: handler.generate_schema(m.Owner))]`.
The metadata delegates schema construction; it does not reconstruct validated
instances. The protocol and model must be structurally compatible. This does not
make unrelated implementations of the protocol valid model inputs: instance reuse,
subclass serialization and validation remain governed by the selected model preset.

## Base Model + Field

```python
from __future__ import annotations

from typing import Annotated

from flext_core import m


class UserModel(m.BaseModel):
    """User payload with annotated name and email fields."""

    name: Annotated[str, m.Field(description="User name")]
    email: Annotated[str, m.Field(description="User email")]


user = UserModel(name="Alice", email="alice@example.com")
expected_name = "Alice"
if user.name != expected_name:
    message = "Unexpected annotated field value"
    raise RuntimeError(message)
```

## ConfigDict + model_dump

```python
from __future__ import annotations

from flext_core import m


class SettingsModel(m.BaseModel):
    """Settings model that ignores extra keys."""

    model_config = m.ConfigDict(extra="ignore")
    debug: bool = False


settings = SettingsModel(debug=True)
data = settings.model_dump()
if data["debug"] is not True:
    message = "Unexpected dumped debug value"
    raise RuntimeError(message)
```

## field_validator

```python
from __future__ import annotations

from typing import Annotated

from flext_core import c, m


class PortModel(m.BaseModel):
    """Model validating its port field against library bounds."""

    port: Annotated[int, m.Field(description="TCP port")]

    @m.field_validator("port")
    @classmethod
    def validate_port(cls, value: int) -> int:
        """Reject ports outside the library bounds.

        Returns:
            The validated ``int``.

        Raises:
            ValueError: When the port is outside the library bounds.

        """
        message = "invalid_port"
        if value < c.MIN_PORT or value > c.MAX_PORT:
            raise ValueError(message)
        return value


valid = PortModel(port=8080)
expected_port = 8080
if valid.port != expected_port:
    message = "Unexpected validated port value"
    raise RuntimeError(message)
```

## examples-backed sanity check

```python
import io
from contextlib import redirect_stdout

from examples.ex_02_flext_settings import ExamplesFlextSettingsUsage

stream = io.StringIO()
with redirect_stdout(stream):
    ExamplesFlextSettingsUsage("docs/guides/pydantic-v2-patterns.md").exercise()
```
