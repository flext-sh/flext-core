"""Runtime bootstrap protocol composed by the context facade."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # mro-wkii.17.26 (codex): reverse p/t edges are annotation-only while the
    # public protocol facade is still being composed.
    from flext_core import p, t


class FlextProtocolsContextBootstrap:
    """Runtime bootstrap option contracts."""

    class RuntimeBootstrapOptions(Protocol):
        """Runtime bootstrap options for service initialization."""

        settings: p.Settings | None
        settings_type: t.SettingsClass | None
        settings_overrides: t.ScalarMapping | None
        context: p.Context | None
        dispatcher: p.Dispatcher | None
        registry: p.Registry | None
        subproject: str | None
        services: t.MappingKV[str, t.RegisterableService] | None
        factories: t.MappingKV[str, t.FactoryCallable] | None
        resources: t.MappingKV[str, t.ResourceCallable] | None
        container_overrides: t.ScalarMapping | None
        wire_modules: t.SequenceOf[ModuleType | str] | None
        wire_packages: t.StrSequence | None
        wire_classes: t.SequenceOf[type] | None


__all__: tuple[str, ...] = ("FlextProtocolsContextBootstrap",)
