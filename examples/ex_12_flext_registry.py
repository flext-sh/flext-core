"""Golden-file example for the registry DSL public APIs."""

from __future__ import annotations

from pathlib import Path
from typing import override

from .ex_12_registry_plugins import Ex12RegistryPlugins


class Ex12RegistryDsl(Ex12RegistryPlugins):
    """Exercise the canonical registry DSL public API."""

    @override
    def exercise(self) -> None:
        """Run all registry DSL example sections."""
        registry, dispatcher = self._exercise_create_and_service_methods()
        self._exercise_summary_and_mixins()
        handler_a, handler_b = self._exercise_registration_and_dispatch(
            registry, dispatcher
        )
        self._exercise_bindings_and_plugin_apis(registry, handler_a, handler_b)
        self._exercise_register_method_and_tracking(registry)


if __name__ == "__main__":
    Ex12RegistryDsl(caller_file=Path(__file__)).run()
