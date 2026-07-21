# flext-core: Quick Reference Map

**File Structure**:
```
src/flext_core/
├── result.py           # r[T] Railway carrier
├── container.py        # FlextContainer DI singleton
├── dispatcher.py       # FlextDispatcher CQRS router
├── settings.py         # FlextSettings config base
├── context.py          # FlextContext request context
├── loggings.py         # FlextLogger structlog bridge
├── runtime.py          # Runtime utilities
├── decorators.py       # @inject, @service decorators
├── handlers.py         # Handler protocols
├── service.py          # s[T] service base class
├── models.py           # m.* models facade
├── constants.py        # c.* constants facade
├── protocols.py        # p.* contracts facade
├── typings.py          # t.* type aliases (+PEP 695)
├── utilities.py        # u.* helpers facade
├── exceptions.py       # e.* domain exceptions
├── registry.py         # Handler registry internals
└── _models/, _utilities/, _protocols/ # Private impl
```

## Quick Imports

```python
from flext_core import r, c, m, t, p, u, e, s, d, h
# or full facades:
from flext_core import FlextResult, FlextContainer, FlextDispatcher, ...
```

**Last Updated**: 2026-04-14