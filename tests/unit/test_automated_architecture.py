"""Architecture validation tests for flext-core using tv (tv)."""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_tests import tm, tv

_SRC = Path(__file__).resolve().parents[2] / "src" / "flext_core"
_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


class TestAutomatedArchitecture:
    """Architecture compliance tests using tv validator."""

    def test_imports_no_violations(self) -> None:
        result = tv.imports(_SRC)
        tm.ok(result)
        tm.that(isinstance(result.value.passed, bool), eq=True)
        tm.that(isinstance(result.value.violations, list), eq=True)

    def test_types_no_violations(self) -> None:
        result = tv.types(_SRC)
        tm.ok(result)
        tm.that(isinstance(result.value.passed, bool), eq=True)
        tm.that(isinstance(result.value.violations, list), eq=True)

    def test_bypass_no_violations(self) -> None:
        result = tv.bypass(_SRC)
        tm.ok(result)
        tm.that(isinstance(result.value.passed, bool), eq=True)
        tm.that(isinstance(result.value.violations, list), eq=True)

    def test_layer_no_cross_layer_imports(self) -> None:
        result = tv.layer(_SRC)
        tm.ok(result)
        tm.that(isinstance(result.value.passed, bool), eq=True)
        tm.that(isinstance(result.value.violations, list), eq=True)

    def test_config_valid(self) -> None:
        result = tv.validate_config(_PYPROJECT)
        tm.ok(result)
        tm.that(isinstance(result.value.passed, bool), eq=True)
        tm.that(isinstance(result.value.violations, list), eq=True)

    @pytest.mark.parametrize("validator", ["imports", "types", "bypass", "layer"])
    def test_all_validators_return_scan_result(self, validator: str) -> None:
        func = getattr(tv, validator)
        result = func(_SRC)
        tm.ok(result)
        tm.that(hasattr(result.value, "passed"), eq=True)
        tm.that(hasattr(result.value, "violations"), eq=True)
        tm.that(hasattr(result.value, "validator_name"), eq=True)
        tm.that(hasattr(result.value, "files_scanned"), eq=True)

    def test_full_validation(self) -> None:
        result = tv.all(_SRC, pyproject_path=_PYPROJECT)
        tm.ok(result)
        tm.that(isinstance(result.value.passed, bool), eq=True)
        tm.that(isinstance(result.value.violations, list), eq=True)
