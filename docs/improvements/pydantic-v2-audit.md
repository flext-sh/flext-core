# Pydantic v2 Audit (Current)

<!-- TOC START -->
- [Summary](#summary)
- [Check: model_dump usage](#check-modeldump-usage)
- [Check: ConfigDict usage](#check-configdict-usage)
- [Check: examples-backed settings flow](#check-examples-backed-settings-flow)
<!-- TOC END -->

## Summary

This audit page keeps only executable checks aligned to the current codebase.

## Check: model_dump usage

```python
from flext_core import m


class AuditModel(m.BaseModel):
    value: int


model = AuditModel(value=1)
data = model.model_dump()
assert data["value"] == 1
```

## Check: ConfigDict usage

```python
from flext_core import m


class AuditSettings(m.BaseModel):
    model_config = m.ConfigDict(extra="ignore")
    debug: bool = False


assert AuditSettings(debug=True).debug is True
```

## Check: examples-backed settings flow

```python
from examples.ex_02_flext_settings import Ex02FlextSettings

Ex02FlextSettings("docs/improvements/pydantic-v2-audit.md").exercise()
```
