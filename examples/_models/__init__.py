# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".errors": ("ExamplesFlextCoreModelsErrors",),
        ".ex00": ("ExamplesFlextCoreModelsEx00",),
        ".ex01": ("ExamplesFlextCoreModelsEx01",),
        ".ex02": (
            "ExamplesFlextCoreModelsEx02",
            "ExamplesFlextCoreSettingsEx02TestConfig",
        ),
        ".ex03": (
            "Ex03Email",
            "Ex03Money",
            "Ex03Order",
            "Ex03OrderItem",
            "Ex03User",
            "ExamplesFlextCoreModelsEx03",
        ),
        ".ex04": ("ExamplesFlextCoreModelsEx04",),
        ".ex05": ("ExamplesFlextCoreModelsEx05",),
        ".ex07": ("ExamplesFlextCoreModelsEx07",),
        ".ex08": (
            "Ex08Order",
            "Ex08User",
            "ExamplesFlextCoreModelsEx08",
        ),
        ".ex10": (
            "Ex10CommandBusStub",
            "Ex10ContextPayload",
            "Ex10DerivedMessage",
            "Ex10Entity",
            "Ex10Message",
            "Ex10ProcessorBad",
            "Ex10ProcessorGood",
            "Ex10ProtocolHandler",
            "Ex10ServiceStub",
            "ExamplesFlextCoreModelsEx10",
        ),
        ".ex11": (
            "Ex11CommandBusStub",
            "Ex11EntityStub",
            "Ex11HandlerLike",
            "Ex11HandlerLikeService",
            "Ex11Payload",
            "Ex11ProcessorProtocolBad",
            "Ex11ProcessorProtocolGood",
            "ExamplesFlextCoreModelsEx11",
        ),
        ".ex12": (
            "Ex12CommandA",
            "Ex12CommandB",
            "ExamplesFlextCoreModelsEx12",
        ),
        ".ex14": ("ExamplesFlextCoreModelsEx14",),
        ".exsettings": ("ExSettingsAppSettings",),
        ".output": ("ExamplesFlextCoreModelsOutput",),
        ".shared": (
            "ExamplesFlextCoreSharedHandle",
            "ExamplesFlextCoreSharedPerson",
        ),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
