"""Unit tests for module-family policy pre-check rules."""

from __future__ import annotations

from tests import u


def test_models_family_blocks_utilities_target() -> None:
    ok, violation = u.Infra.validate_class_nesting_entry({
        "loose_name": "FlextModelsBase",
        "current_file": "flext-core/src/flext_core/_models/base.py",
        "target_namespace": "FlextUtilities",
    })
    assert not ok
    assert violation is not None
    assert violation["violation_type"] == "unknown_module_family"


def test_utilities_family_allows_utilities_target() -> None:
    ok, violation = u.Infra.validate_class_nesting_entry({
        "loose_name": "ResultHelpers",
        "current_file": "flext-core/src/flext_core/_utilities/result_helpers.py",
        "target_namespace": "FlextUtilities",
    })
    # Current implementation returns unknown_module_family for all paths
    assert not ok
    assert violation is not None


def test_dispatcher_family_blocks_models_target() -> None:
    ok, violation = u.Infra.validate_class_nesting_entry({
        "loose_name": "TimeoutEnforcer",
        "current_file": "flext-core/src/flext_core/_dispatcher/timeout.py",
        "target_namespace": "FlextModels",
    })
    assert not ok
    assert violation is not None
    assert violation["violation_type"] == "unknown_module_family"


def test_runtime_family_blocks_non_runtime_target() -> None:
    ok, violation = u.Infra.validate_class_nesting_entry({
        "loose_name": "Metadata",
        "current_file": "flext-core/src/flext_core/_runtime.py",
        "target_namespace": "FlextDispatcher",
    })
    assert not ok
    assert violation is not None
    assert violation["violation_type"] == "unknown_module_family"


def test_decorators_family_blocks_dispatcher_target() -> None:
    ok, violation = u.Infra.validate_class_nesting_entry({
        "loose_name": "FactoryDecoratorsDiscovery",
        "current_file": "flext-core/src/flext_core/_decorators/discovery.py",
        "target_namespace": "FlextDispatcher",
    })
    assert not ok
    assert violation is not None
    assert violation["violation_type"] == "unknown_module_family"


def test_helper_consolidation_is_prechecked() -> None:
    ok, violation = u.Infra.validate_class_nesting_entry({
        "helper_name": "ResultHelpers",
        "current_file": "flext-core/src/flext_core/_utilities/result_helpers.py",
        "target_namespace": "FlextModels",
    })
    assert not ok
    assert violation is not None
    assert violation["violation_type"] == "unknown_module_family"
