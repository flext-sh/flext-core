# Pydantic v2 Patterns

<!-- TOC START -->

- [Overview](#overview)
- [Base Model + Field](#base-model-field)
- [ConfigDict + model\_dump](#configdict-model_dump)
- [field\_validator](#field_validator)
- [examples-backed sanity check](#examples-backed-sanity-check)

<!-- TOC END -->

## Overview

This guide shows practical Pydantic v2 patterns used in `flext-core`.

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

from flext_core import c, m, u


class PortModel(m.BaseModel):
    """Model validating its port field against library bounds."""

    port: Annotated[int, m.Field(description="TCP port")]

    @u.field_validator("port")
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

from examples.ex_02_flext_settings import Ex02FlextSettings

stream = io.StringIO()
with redirect_stdout(stream):
    Ex02FlextSettings("docs/guides/pydantic-v2-patterns.md").exercise()
```
