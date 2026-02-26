from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast, runtime_checkable

import pytest

from flext_core import c, m, p, r, t, u


@runtime_checkable
class _NamedProtocol(Protocol):
    def _protocol_name(self) -> str: ...


class _SettingsModel(p.ProtocolSettings, _NamedProtocol):
    app_name: str = "x"

    def _protocol_name(self) -> str:
        return "settings"


class _ProtocolModel(p.ProtocolModel, _NamedProtocol):
    name: str = "ok"

    def _protocol_name(self) -> str:
        return "model"


@runtime_checkable
class _RequirePing(Protocol):
    def ping(self) -> str: ...


def test_implements_decorator_validation_error_message() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert r[int].ok(1).is_success
    assert isinstance(m.ConfigMap(root={}), m.ConfigMap)
    assert isinstance(u.Conversion.to_str_list(1), list)

    with pytest.raises(TypeError, match="does not implement required members"):

        @p.implements(_RequirePing)
        class _Invalid:
            pass


def test_protocol_meta_default_model_base_and_get_protocols_default() -> None:
    class _MetaCreated(_NamedProtocol, metaclass=p.ProtocolModelMeta):
        def _protocol_name(self) -> str:
            return "meta"

    instance = _MetaCreated()
    assert isinstance(instance, _MetaCreated)
    assert getattr(_MetaCreated, "__protocols__", ()) == (_NamedProtocol,)

    class _NoProtocols(m.ArbitraryTypesModel, metaclass=p.ProtocolModelMeta):
        x: int = 1

    assert not hasattr(_NoProtocols, "get_protocols")
    assert getattr(_NoProtocols, "__protocols__", ()) == ()


def test_protocol_model_and_settings_methods() -> None:
    model = _ProtocolModel(name="n")
    settings = _SettingsModel(app_name="a")

    assert model.implements_protocol(_NamedProtocol) is True
    assert model.get_protocols() == (_NamedProtocol,)
    assert model._protocol_name() == "model"

    assert settings.implements_protocol(_NamedProtocol) is True
    assert settings.get_protocols() == (_NamedProtocol,)
    assert settings._protocol_name() == "settings"


def test_implements_decorator_helper_methods_and_static_wrappers() -> None:
    @p.implements(_NamedProtocol)
    class _Decorated:
        def _protocol_name(self) -> str:
            return "decorated"

    obj = _Decorated()
    implements = cast(Callable[[type], bool], getattr(obj, "implements_protocol"))
    get_protocols = cast(
        Callable[[], tuple[type, ...]], getattr(_Decorated, "get_protocols")
    )
    assert implements(_NamedProtocol) is True
    assert get_protocols() == (_NamedProtocol,)
    assert p.is_protocol(_NamedProtocol) is True
    assert (
        p.check_implements_protocol(cast(t.GuardInputValue, obj), _NamedProtocol)
        is True
    )


def test_check_implements_protocol_false_non_runtime_protocol() -> None:
    class _NotAProtocol:
        pass

    @p.implements(_NamedProtocol)
    class _Thing:
        def _protocol_name(self) -> str:
            return "thing"

    obj = _Thing()
    assert (
        p.check_implements_protocol(cast(t.GuardInputValue, obj), _NotAProtocol)
        is False
    )


def test_protocol_base_name_methods_and_runtime_check_branch() -> None:
    class _OnlyRuntime:
        def _protocol_name(self) -> str:
            return "runtime"

    runtime_obj = _OnlyRuntime()
    assert (
        p.check_implements_protocol(
            cast(t.GuardInputValue, runtime_obj), _NamedProtocol
        )
        is True
    )

    class _DefaultModelName(p.ProtocolModel):
        value: int = 1

    class _DefaultSettingsName(p.ProtocolModel):
        app_name: str = "x"

    model_name_getter = cast(
        Callable[[], str],
        getattr(_DefaultModelName(), "_protocol_name"),
    )
    settings_name_getter = cast(
        Callable[[], str],
        getattr(_DefaultSettingsName(), "_protocol_name"),
    )
    assert model_name_getter() == "_DefaultModelName"
    assert settings_name_getter() == "_DefaultSettingsName"
