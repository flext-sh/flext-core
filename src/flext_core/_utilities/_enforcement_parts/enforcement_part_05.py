"""Runtime enforcement engine MRO part."""

from __future__ import annotations

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._constants.regex import FlextConstantsRegex as cre

from .enforcement_part_04 import (
    FlextUtilitiesEnforcement as FlextUtilitiesEnforcementPart04,
)


class FlextUtilitiesEnforcement(FlextUtilitiesEnforcementPart04):
    @staticmethod
    def class_name_to_module(class_name: str) -> str:
        """Map a ``Flext<Project><Layer><Concern>`` class to its owning package.

        SSOT for the convention: facade-layer classes (one of
        ``c.NAMESPACE_LAYER_NAMES``) are re-exported from the project's
        top-level package, so ``FlextCliUtilitiesAuth`` is imported from
        ``flext_cli`` — never from the synthetic ``flext_cli_utilities_auth``
        path produced by a naïve CamelCase-to-snake_case conversion.

        Used by both detection (enforcement rules that flag a wrong import
        path on a facade-layer class) and correction (refactor verbs that
        emit the right ``from flext_<project> import <Class>`` line).

        Minimal exceptions to the project-prefix convention live in
        ``c.NAMESPACE_CLASS_TO_MODULE_OVERRIDES``. Inputs that match
        neither the override table nor the project/layer pattern are
        a contract violation — the function raises ``ValueError`` with
        the offending class name.
        """
        override = c.NAMESPACE_CLASS_TO_MODULE_OVERRIDES.get(class_name)
        if override is not None:
            return override
        flext_prefix = "Flext"
        if not class_name.startswith(flext_prefix):
            msg = (
                f"class_name_to_module: {class_name!r} is not a "
                f"Flext-prefixed facade class and has no override in "
                f"c.NAMESPACE_CLASS_TO_MODULE_OVERRIDES"
            )
            raise ValueError(msg)
        tail = class_name[len(flext_prefix) :]
        for layer in c.NAMESPACE_LAYER_NAMES:
            idx = tail.find(layer)
            if idx > 0:
                project = tail[:idx]
                snake = cre.CAMEL_TO_SNAKE_RE.sub(r"\1_\2", project).lower()
                return f"flext_{snake}"
        msg = (
            f"class_name_to_module: {class_name!r} contains no facade "
            f"layer suffix from c.NAMESPACE_LAYER_NAMES "
            f"({tuple(c.NAMESPACE_LAYER_NAMES)}) and has no override "
            f"in c.NAMESPACE_CLASS_TO_MODULE_OVERRIDES"
        )
        raise ValueError(msg)


__all__: list[str] = ["FlextUtilitiesEnforcement"]
