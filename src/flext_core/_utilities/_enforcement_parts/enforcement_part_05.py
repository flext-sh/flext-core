"""Runtime enforcement engine MRO part.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._constants.regex import FlextConstantsRegex as cre
from flext_core._utilities._enforcement_parts.enforcement_part_03 import (
    FlextUtilitiesEnforcement as FlextUtilitiesEnforcementPart03,
)


class FlextUtilitiesEnforcement(FlextUtilitiesEnforcementPart03):
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

        Inputs that do not match the project/layer pattern are a contract
        violation — the function raises ``ValueError`` with the offending
        class name.

        Returns:
            The resulting ``str``.

        Raises:
            ValueError: If class_name_to_module.

        """
        flext_prefix = "Flext"
        if not class_name.startswith(flext_prefix):
            msg = (
                f"class_name_to_module: {class_name!r} is not a "
                f"Flext-prefixed facade class"
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
            f"({tuple(c.NAMESPACE_LAYER_NAMES)})"
        )
        raise ValueError(msg)


__all__: list[str] = ["FlextUtilitiesEnforcement"]
