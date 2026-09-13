"""The module-callable walk never resolves a foreign lazy proxy.

Why this exists: flext-observability imports Flask, Flask publishes two
module-level ``LocalProxy`` objects, and the runtime-census gate walked them.
A ``LocalProxy`` forwards every attribute, ``__class__`` included, and raises
outside an application context, so the gate died with the proxy's own error
and produced no findings report at all.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Never

import pytest

from flext_core import u

_OUTSIDE_CONTEXT = "Working outside of application context"


class _ForwardingProxy:
    """A stand-in with werkzeug ``LocalProxy``'s forwarding behaviour."""

    @property
    def __class__(self) -> type:
        """Forward the type question to an object that is not there."""
        raise RuntimeError(_OUTSIDE_CONTEXT)

    def __getattr__(self, name: str) -> Never:
        """Forward every attribute to an object that is not there."""
        raise RuntimeError(_OUTSIDE_CONTEXT)


def _defined_here() -> int:
    """Stand for a real function the module under test owns."""
    return 1


class TestsFlextCoreBeartypeModuleCallables:
    """Behaviour of the module-callable walk over a hostile namespace."""

    @staticmethod
    @contextmanager
    def _module_with_proxy() -> Iterator[types.ModuleType]:
        """Publish a module whose namespace holds a function and a proxy.

        The walk keeps only functions the module itself defines, and that
        question is answered through the import system, so the module has to
        be reachable there while the case runs.
        """
        module = types.ModuleType("probe_module_with_proxy")
        original_module_name = _defined_here.__module__
        _defined_here.__module__ = module.__name__
        module.__dict__["defined_here"] = _defined_here
        module.__dict__["current_app"] = _ForwardingProxy()
        sys.modules[module.__name__] = module
        try:
            yield module
        finally:
            del sys.modules[module.__name__]
            _defined_here.__module__ = original_module_name

    def test_proxy_is_skipped_and_the_real_function_is_yielded(self) -> None:
        """The walk completes, ignoring the proxy and keeping the function."""
        with TestsFlextCoreBeartypeModuleCallables._module_with_proxy() as module:
            yielded = [
                function.__name__ for function in u.iter_module_callables(module)
            ]

        assert yielded == [_defined_here.__name__]

    def test_the_proxy_really_would_raise_on_resolution(self) -> None:
        """The stand-in is faithful: resolving it raises, as the real one does.

        Without this, the case above could pass against a proxy that quietly
        answers, and would no longer prove anything.
        """
        proxy = _ForwardingProxy()

        with pytest.raises(RuntimeError, match=_OUTSIDE_CONTEXT):
            isinstance(proxy, classmethod)

        with pytest.raises(RuntimeError, match=_OUTSIDE_CONTEXT):
            _ = proxy.__code__


__all__: list[str] = ["TestsFlextCoreBeartypeModuleCallables"]
