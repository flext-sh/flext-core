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
from typing import TYPE_CHECKING, Never

import pytest

from flext_core import u

if TYPE_CHECKING:
    from flext_core import p

_OUTSIDE_CONTEXT = "Working outside of application context"
_PROBE_MODULE = "probe_module_with_proxy"


def _defined_here() -> int:
    """Stand for a real function the module under test owns."""
    return 1


class TestsFlextCoreBeartypeModuleCallables:
    """Behaviour of the module-callable walk over a hostile namespace."""

    class _ForwardingProxy:
        """A stand-in with werkzeug ``LocalProxy``'s forwarding behaviour.

        ``__class__`` accepts any class through the canonical attribute-probe
        contract, preserving the writable ``object`` interface while forwarding
        the question the way the real proxy does.
        """

        @property
        def __class__(
            self,
        ) -> type[TestsFlextCoreBeartypeModuleCallables._ForwardingProxy]:
            """Forward the type question to an object that is not there."""
            raise RuntimeError(_OUTSIDE_CONTEXT)

        @__class__.setter
        def __class__(self, value: type[p.AttributeProbe]) -> None:
            """Forward the assignment too, so the override stays read-write."""
            raise RuntimeError(_OUTSIDE_CONTEXT)

        def __getattr__(self, name: str) -> Never:
            """Forward every attribute to an object that is not there."""
            raise RuntimeError(_OUTSIDE_CONTEXT)

    def test_proxy_is_skipped_and_the_real_function_is_yielded(self) -> None:
        """The walk completes, ignoring the proxy and keeping the function.

        The walk keeps only functions the module itself defines, and answers
        that question through the import system, so the probe module has to be
        reachable there while the case runs.
        """
        module = types.ModuleType(_PROBE_MODULE)
        original_module_name = _defined_here.__module__
        _defined_here.__module__ = _PROBE_MODULE
        module.__dict__["defined_here"] = _defined_here
        module.__dict__["current_app"] = (
            TestsFlextCoreBeartypeModuleCallables._ForwardingProxy()
        )
        sys.modules[_PROBE_MODULE] = module
        try:
            yielded = [
                function.__name__ for function in u.iter_module_callables(module)
            ]
        finally:
            del sys.modules[_PROBE_MODULE]
            _defined_here.__module__ = original_module_name

        assert yielded == [_defined_here.__name__]

    def test_the_proxy_really_would_raise_on_resolution(self) -> None:
        """The stand-in is faithful: resolving it raises, as the real one does.

        Without this, the case above could pass against a proxy that quietly
        answers, and would no longer prove anything.
        """
        proxy = TestsFlextCoreBeartypeModuleCallables._ForwardingProxy()

        with pytest.raises(RuntimeError, match=_OUTSIDE_CONTEXT):
            isinstance(proxy, classmethod)

        with pytest.raises(RuntimeError, match=_OUTSIDE_CONTEXT):
            _ = proxy.__code__

        with pytest.raises(RuntimeError, match=_OUTSIDE_CONTEXT):
            proxy.__class__ = object


__all__: list[str] = ["TestsFlextCoreBeartypeModuleCallables"]
