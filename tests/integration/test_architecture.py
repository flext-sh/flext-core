"""Architecture validation tests for flext-core using tv (tv)."""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_tests import tm, tv

from tests import c


class TestsFlextAutomatedArchitecture:
    """Architecture compliance tests using tv validator."""

    def test_imports_no_violations(self) -> None:
        result = tv.imports(
            Path(__file__).resolve().parents[c.Tests.REPO_ROOT_PARENT_DEPTH]
            / c.Tests.SRC_DIR
            / c.Tests.CORE_PACKAGE_DIR,
        )
        tm.ok(result)
        tm.that(result.value.passed, is_=bool)
        tm.that(result.value.violations, is_=list)

    def test_types_no_violations(self) -> None:
        result = tv.types(
            Path(__file__).resolve().parents[c.Tests.REPO_ROOT_PARENT_DEPTH]
            / c.Tests.SRC_DIR
            / c.Tests.CORE_PACKAGE_DIR,
        )
        tm.ok(result)
        tm.that(result.value.passed, is_=bool)
        tm.that(result.value.violations, is_=list)

    def test_bypass_no_violations(self) -> None:
        result = tv.bypass(
            Path(__file__).resolve().parents[c.Tests.REPO_ROOT_PARENT_DEPTH]
            / c.Tests.SRC_DIR
            / c.Tests.CORE_PACKAGE_DIR,
        )
        tm.ok(result)
        tm.that(result.value.passed, is_=bool)
        tm.that(result.value.violations, is_=list)

    def test_layer_no_cross_layer_imports(self) -> None:
        result = tv.layer(
            Path(__file__).resolve().parents[c.Tests.REPO_ROOT_PARENT_DEPTH]
            / c.Tests.SRC_DIR
            / c.Tests.CORE_PACKAGE_DIR,
        )
        tm.ok(result)
        tm.that(result.value.passed, is_=bool)
        tm.that(result.value.violations, is_=list)

    def test_config_valid(self) -> None:
        result = tv.validate_config(
            Path(__file__).resolve().parents[c.Tests.REPO_ROOT_PARENT_DEPTH]
            / c.Tests.PYPROJECT_FILENAME,
        )
        tm.ok(result)
        tm.that(result.value.passed, is_=bool)
        tm.that(result.value.violations, is_=list)

    @pytest.mark.parametrize(
        "validator",
        c.Tests.VALIDATOR_METHODS,
    )
    def test_all_validators_return_scan_result(self, validator: str) -> None:
        func = getattr(tv, validator)
        result = func(
            Path(__file__).resolve().parents[c.Tests.REPO_ROOT_PARENT_DEPTH]
            / c.Tests.SRC_DIR
            / c.Tests.CORE_PACKAGE_DIR,
        )
        tm.ok(result)

    def test_full_validation(self) -> None:
        repo_root = Path(__file__).resolve().parents[c.Tests.REPO_ROOT_PARENT_DEPTH]
        result = tv.all(
            repo_root / c.Tests.SRC_DIR / c.Tests.CORE_PACKAGE_DIR,
            options=tv.AllValidationOptions(
                pyproject_path=repo_root / c.Tests.PYPROJECT_FILENAME,
            ),
        )
        tm.ok(result)
        tm.that(result.value.passed, is_=bool)
        tm.that(result.value.violations, is_=list)
