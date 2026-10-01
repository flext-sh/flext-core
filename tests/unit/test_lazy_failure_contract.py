"""The lazy resolver fails loud on broken modules and stays probe-safe on absence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from flext_tests import tm

from flext_core.lazy import lazy_getattr

if TYPE_CHECKING:
    from pathlib import Path


class TestsFlextCoreLazyFailureContract:
    """Drive the public lazy resolver against real modules."""

    def test_broken_target_module_raises_import_error_with_its_cause(
        self,
        tmp_path: Path,
    ) -> None:
        """A target module whose body fails is a defect, never a missing name."""
        package = f"lazy_contract_{tmp_path.name.replace('-', '_')}"
        root = tmp_path / package
        root.mkdir()
        (root / "__init__.py").write_text("", encoding="utf-8")
        (root / "broken.py").write_text(
            'msg = "stale constants body"\nraise AttributeError(msg)\n',
            encoding="utf-8",
        )
        target = f"{package}.broken"
        with (
            tm.scope(python_paths=[str(tmp_path)]),
            pytest.raises(ImportError, match="lazy import of") as caught,
        ):
            lazy_getattr("broken", {"broken": target}, {}, package)
        assert isinstance(caught.value.__cause__, AttributeError)
        tm.that(str(caught.value), has=target)

    def test_absent_symbol_on_a_loaded_module_stays_an_attribute_error(self) -> None:
        """A legal absence keeps hasattr and getattr-with-default probes working."""
        with pytest.raises(AttributeError, match="has no attribute"):
            lazy_getattr(
                "Missing",
                {"Missing": ("xml.dom", "DoesNotExist")},
                {},
                "xml.etree",
            )
