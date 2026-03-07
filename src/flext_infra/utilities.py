"""Utilities facade for flext-infra.

Re-exports flext_core utilities and adds infrastructure-specific
utility namespaces for terminal, I/O, YAML, codegen, and refactor helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextUtilities
from flext_infra._utilities.io import FlextInfraUtilitiesIo
from flext_infra._utilities.terminal import FlextInfraUtilitiesTerminal
from flext_infra._utilities.toml import FlextInfraUtilitiesToml
from flext_infra._utilities.yaml import FlextInfraUtilitiesYaml
from flext_infra.codegen._utilities import FlextInfraUtilitiesCodegen
from flext_infra.refactor._utilities import FlextInfraUtilitiesRefactor


class FlextInfraUtilities(FlextUtilities):
    """Utility namespace for flext-infra; extends FlextUtilities.

    Usage::

        from flext_infra import u

        # Core utilities (inherited)
        u.generate()
        u.parse(value, int)

        # Infra-specific utilities
        u.Infra.Terminal.should_use_color()
        u.Infra.Io.read_json(path)
        u.Infra.Yaml.safe_load_yaml(path)
        u.Infra.Codegen.infer_package(path)
        u.Infra.Refactor.dotted_name(expr)
        u.Infra.Toml.as_toml_mapping(value)
    """

    class Infra:
        """Infrastructure-domain utilities."""

        class Terminal(FlextInfraUtilitiesTerminal):
            """Terminal capability detection — real inheritance."""

        class Io(FlextInfraUtilitiesIo):
            """I/O convenience helpers — real inheritance."""

        class Yaml(FlextInfraUtilitiesYaml):
            """YAML loading and validation — real inheritance."""

        class Codegen(FlextInfraUtilitiesCodegen):
            """Code generation and AST helpers — real inheritance."""

        class Refactor(FlextInfraUtilitiesRefactor):
            """CST/refactor analysis helpers — real inheritance."""

        class Toml(FlextInfraUtilitiesToml):
            """TOML type-safe mapping narrowing — real inheritance."""


u = FlextInfraUtilities

__all__ = ["FlextInfraUtilities", "u"]
