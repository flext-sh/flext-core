# Documentation Standards


<!-- TOC START -->
- [Requirements](#requirements)
- [Executable Standard Snippet](#executable-standard-snippet)
- [Examples-backed Reference](#examples-backed-reference)
<!-- TOC END -->

## Requirements

- Snippets must be executable or explicitly marked as `text`.
- Snippets should be self-contained.
- Prefer examples-backed references for behavior coverage.

## Executable Standard Snippet

```python
from flext_core import FlextSettings

settings = FlextSettings.fetch_global()
assert isinstance(settings.model_dump(), dict)
```

## Examples-backed Reference

```python
from examples.ex_02_flext_settings import Ex02FlextSettings

Ex02FlextSettings("docs/standards/documentation.md").exercise()
```
