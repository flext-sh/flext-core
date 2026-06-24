# Pydantic v2 Audit (Current)

## Summary

This audit page keeps only executable checks aligned to the current codebase.

## Check: model_dump usage

```python
from __future__ import annotations

from flext_core import m


class AuditModel(m.BaseModel):
    value: int


model = AuditModel(value=1)
data = model.model_dump()
assert data["value"] == 1
```

## Check: ConfigDict usage

```python
from __future__ import annotations

from flext_core import m


class AuditSettings(m.BaseModel):
    model_config = m.ConfigDict(extra="ignore")
    debug: bool = False


assert AuditSettings(debug=True).debug is True
```

## Check: examples-backed settings flow

```python
import io
from contextlib import redirect_stdout

from examples.ex_02_flext_settings import Ex02FlextSettings

stream = io.StringIO()
with redirect_stdout(stream):
    Ex02FlextSettings("docs/improvements/pydantic-v2-audit.md").exercise()
```
