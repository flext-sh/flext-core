# Pydantic v2 Patterns

## Overview

This guide shows practical Pydantic v2 patterns used in `flext-core`.

## Base Model + Field

```python
from typing import Annotated

from flext_core import m


class UserModel(m.BaseModel):
    name: Annotated[str, m.Field(description="User name")]
    email: Annotated[str, m.Field(description="User email")]


user = UserModel(name="Alice", email="alice@example.com")
assert user.name == "Alice"
```

## ConfigDict + model_dump

```python
from flext_core import m


class SettingsModel(m.BaseModel):
    model_config = m.ConfigDict(extra="ignore")
    debug: bool = False


settings = SettingsModel(debug=True)
data = settings.model_dump()
assert data["debug"] is True
```

## field_validator

```python
from typing import Annotated

from flext_core import m, u


class PortModel(m.BaseModel):
    port: Annotated[int, m.Field(description="TCP port")]

    @u.field_validator("port")
    @classmethod
    def validate_port(cls, value: int) -> int:
        if value < 1 or value > 65535:
            raise ValueError("invalid_port")
        return value


valid = PortModel(port=8080)
assert valid.port == 8080
```

## examples-backed sanity check

```python
from examples.ex_02_flext_settings import Ex02FlextSettings

Ex02FlextSettings("docs/guides/pydantic-v2-patterns.md").exercise()
```
