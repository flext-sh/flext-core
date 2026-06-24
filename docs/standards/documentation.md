# Documentation Standards

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
import io
from contextlib import redirect_stdout

from examples.ex_02_flext_settings import Ex02FlextSettings

stream = io.StringIO()
with redirect_stdout(stream):
    Ex02FlextSettings("docs/standards/documentation.md").exercise()
```
