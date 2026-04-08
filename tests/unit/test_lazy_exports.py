"""Tests for module-level lazy export installation behavior."""

from __future__ import annotations

from flext_core.lazy import install_lazy_exports


class TestInstallLazyExports:
    """Verify __all__ publication stays root-only when requested."""

    def test_publish_all_disabled_omits_all(self) -> None:
        """Subpackages keep __dir__ but do not publish __all__."""
        module_globals: dict[str, object] = {}

        install_lazy_exports(
            "test_pkg.transformers",
            module_globals,
            {"Alpha": ("test_pkg.transformers.alpha", "Alpha")},
            publish_all=False,
        )

        assert "__all__" not in module_globals
        assert module_globals["__dir__"]() == ["Alpha"]

    def test_publish_all_enabled_keeps_all(self) -> None:
        """Root packages still publish __all__ by default."""
        module_globals: dict[str, object] = {}

        install_lazy_exports(
            "test_pkg",
            module_globals,
            {"Alpha": ("test_pkg.alpha", "Alpha")},
        )

        assert module_globals["__all__"] == ("Alpha",)
        assert module_globals["__dir__"]() == ["Alpha"]
