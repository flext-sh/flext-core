"""Tests for FlextProtocols to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast, override, runtime_checkable

from flext_core import c, m, p, r, u


@runtime_checkable
class _RequirePing(Protocol):
    def ping(self) -> str: ...


class _SettingsModel(p.ProtocolSettings, _RequirePing):
    app_name: str = "x"

    @override
    def ping(self) -> str:
        return "pong"


class _ProtocolModel(p.ProtocolModel, _RequirePing):
    name: str = "ok"

    @override
    def ping(self) -> str:
        return "pong"


def test_implements_decorator_validation_error_message() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert r[int].ok(1).is_success
    assert isinstance(m.ConfigMap(root={}), m.ConfigMap)
    assert isinstance(u.to_str_list(1), list)

    @p.implements(_RequirePing)
    class _Invalid:
        pass

    instance = _Invalid()
    assert p.check_implements_protocol(cast("object", instance), _RequirePing) is True


def test_protocol_model_and_settings_methods() -> None:
    model = _ProtocolModel()
    settings = _SettingsModel(app_name="a")
    assert model.implements_protocol(_RequirePing) is True
    assert model.get_protocols() == (_RequirePing,)
    assert settings.implements_protocol(_RequirePing) is True
    assert settings.get_protocols() == (_RequirePing,)


def test_implements_decorator_helper_methods_and_static_wrappers() -> None:

    @p.implements(_RequirePing)
    class _Decorated:
        def ping(self) -> str:
            return "pong"

    obj = _Decorated()
    implements = cast("Callable[[type], bool]", getattr(obj, "implements_protocol"))
    get_protocols = cast(
        "Callable[[], tuple[type, ...]]",
        getattr(_Decorated, "get_protocols"),
    )
    assert implements(_RequirePing) is True
    assert get_protocols() == (_RequirePing,)
    assert p.is_protocol(_RequirePing) is True
    assert p.check_implements_protocol(cast("object", obj), _RequirePing) is True


def test_check_implements_protocol_false_non_runtime_protocol() -> None:

    class _NotA:
        pass

    @p.implements(_RequirePing)
    class _Thing:
        def ping(self) -> str:
            return "pong"

    obj = _Thing()
    assert p.check_implements_protocol(cast("object", obj), _NotA) is False


def test_protocol_runtime_check() -> None:

    class _OnlyRuntime:
        def ping(self) -> str:
            return "pong"

    runtime_obj = _OnlyRuntime()
    assert (
        p.check_implements_protocol(
            cast("object", runtime_obj),
            _RequirePing,
        )
        is True
    )
