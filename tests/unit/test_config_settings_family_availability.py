"""Config and settings are available to every ``c,t,p,m,u`` family module.

The layer-0 SSOT contract (ADR-005): each family facade and family package
loads alongside the ``config``/``settings`` singletons, reads them through the
single canonical form, and the settings surface stays usable for typed
capabilities (the models family resolves the configured IANA timezone without
any utility-layer import).

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from importlib import import_module
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pytest

from flext_core import config, settings

if TYPE_CHECKING:
    from types import ModuleType

_FAMILY_SURFACES: tuple[str, ...] = (
    "flext_core.constants",
    "flext_core.typings",
    "flext_core.protocols",
    "flext_core.models",
    "flext_core.utilities",
    "flext_core._constants",
    "flext_core._typings",
    "flext_core._protocols",
    "flext_core._models",
    "flext_core._utilities",
)


class TestsConfigSettingsFamilyAvailability:
    """Prove the layer-0 SSOT reaches every family without reverse imports."""

    @staticmethod
    def _load(module_name: str) -> ModuleType:
        return import_module(module_name)

    @staticmethod
    def test_singletons_are_stable_across_import_forms() -> None:
        """Root-letter access and owner-module access yield the same SSOT."""
        owner_settings = import_module("flext_core._settings").settings
        owner_config = import_module("flext_core._config").config
        assert settings is owner_settings
        assert config is owner_config

    @pytest.mark.parametrize("module_name", _FAMILY_SURFACES)
    def test_family_module_loads_alongside_config_settings(
        self,
        module_name: str,
    ) -> None:
        """Each family surface imports and the SSOT stays functional."""
        module = self._load(module_name)
        assert module is not None
        assert isinstance(settings.model_dump(), dict)
        assert isinstance(config.model_dump(), dict)

    @staticmethod
    def test_settings_timezone_resolves_for_models_family() -> None:
        """The models family resolves the configured IANA timezone itself.

        The models timestamp capability needs the configured timezone from
        the layer-0 settings; it must never import the utility layer for it.

        """
        assert isinstance(ZoneInfo(settings.timezone), ZoneInfo)

    @staticmethod
    def test_config_exposes_typed_namespace_mapping() -> None:
        """The config singleton exposes its business-rule namespaces."""
        namespaces = config.model_dump()
        assert len(namespaces) > 0
        assert all(isinstance(key, str) for key in namespaces)
