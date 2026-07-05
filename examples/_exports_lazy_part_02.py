# AUTO-GENERATED FILE — Regenerate with: make gen
"""Lazy export map part."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map

EXAMPLES_LAZY_IMPORTS_PART_02 = build_lazy_import_map(
    {
        "._models": ("_models",),
        "._models.ex10": ("ExamplesFlextModelsEx10",),
        "._models.ex11": ("ExamplesFlextModelsEx11",),
        "._models.ex12": ("ExamplesFlextModelsEx12",),
        "._models.ex14": ("ExamplesFlextModelsEx14",),
        "._models.output": ("ExamplesFlextModelsOutput",),
        "._models.shared": (
            "ExamplesFlextSharedHandle",
            "ExamplesFlextSharedPerson",
        ),
        "._shared_parts": ("_shared_parts",),
        "._shared_parts.shared_part_01": ("ExamplesFlextSharedBase",),
        ".constants": ("c",),
        ".ex_12_registry_support": ("ProtocolHandler",),
        ".models": ("m",),
        ".protocols": ("p",),
        ".settings": ("ExamplesSettings",),
        ".shared": ("ExamplesFlextShared",),
        ".typings": (
            "ExamplesFlextTypes",
            "t",
        ),
        ".utilities": ("u",),
        "flext_core._root_typing_parts": (
            "d",
            "e",
            "h",
            "r",
            "s",
            "x",
        ),
    },
)

__all__: list[str] = ["EXAMPLES_LAZY_IMPORTS_PART_02"]
