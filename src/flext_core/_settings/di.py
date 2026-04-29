"""FlextSettingsDI — dependency-injection bridge for settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextConstants as c, FlextTypes as t, FlextUtilities as u


class FlextSettingsDI:
    """Mixin that lazily builds a DI ``Singleton`` provider.

    The provider wraps the current settings instance.
    """

    _di_provider: t.Scalar | None = None

    def resolve_di_settings_provider(self) -> t.Scalar:
        """Get dependency injection provider for this settings.

        Returns a providers.Singleton instance via the runtime bridge.
        Type annotation stays framework-level to avoid DI imports in this module.
        """
        if not hasattr(self, "_di_provider") or self._di_provider is None:
            providers_module = u.dependency_providers()
            self._di_provider = providers_module.Singleton(lambda: self)
        provider = self._di_provider
        if provider is None:
            msg = c.ERR_SETTINGS_DI_PROVIDER_NOT_INITIALIZED
            raise RuntimeError(msg)
        return provider


__all__: list[str] = ["FlextSettingsDI"]
