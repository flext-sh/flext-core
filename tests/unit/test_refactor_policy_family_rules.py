"""Unit tests for module-family policy pre-check rules."""

from __future__ import annotations

from pathlib import Path

from flext_infra import FlextInfraPreCheckGate as PreCheckGate, class_reconstructor


def _policy_path() -> Path:
    return Path(class_reconstructor.__file__).with_name("class-policy-v2.yml")


def test_models_family_blocks_utilities_target() -> None:
    gate = PreCheckGate(policy_path=_policy_path())
    ok, violation = gate.validate_entry({
        "loose_name": "FlextModelFoundation",
        "current_file": "flext-core/src/flext_core/_models/base.py",
        "target_namespace": "FlextUtilities",
    })
    assert not ok
    assert violation is not None
    assert violation["violation_type"] == "forbidden_target"


def test_utilities_family_allows_utilities_target() -> None:
    gate = PreCheckGate(policy_path=_policy_path())
    ok, violation = gate.validate_entry({
        "loose_name": "ResultHelpers",
        "current_file": "flext-core/src/flext_core/_utilities/result_helpers.py",
        "target_namespace": "FlextUtilities",
    })
    assert ok
    assert violation is None


def test_dispatcher_family_blocks_models_target() -> None:
    gate = PreCheckGate(policy_path=_policy_path())
    ok, violation = gate.validate_entry({
        "loose_name": "TimeoutEnforcer",
        "current_file": "flext-core/src/flext_core/_dispatcher/timeout.py",
        "target_namespace": "FlextModels",
    })
    assert not ok
    assert violation is not None
    assert violation["violation_type"] == "forbidden_target"


def test_runtime_family_blocks_non_runtime_target() -> None:
    gate = PreCheckGate(policy_path=_policy_path())
    ok, violation = gate.validate_entry({
        "loose_name": "Metadata",
        "current_file": "flext-core/src/flext_core/_runtime.py",
        "target_namespace": "FlextDispatcher",
    })
    assert not ok
    assert violation is not None
    assert violation["violation_type"] == "forbidden_target"


def test_decorators_family_blocks_dispatcher_target() -> None:
    gate = PreCheckGate(policy_path=_policy_path())
    ok, violation = gate.validate_entry({
        "loose_name": "FactoryDecoratorsDiscovery",
        "current_file": "flext-core/src/flext_core/_decorators/discovery.py",
        "target_namespace": "FlextDispatcher",
    })
    assert not ok
    assert violation is not None
    assert violation["violation_type"] == "forbidden_target"


def test_helper_consolidation_is_prechecked() -> None:
    gate = PreCheckGate(policy_path=_policy_path())
    ok, violation = gate.validate_entry({
        "helper_name": "ResultHelpers",
        "current_file": "flext-core/src/flext_core/_utilities/result_helpers.py",
        "target_namespace": "FlextModels",
    })
    assert not ok
    assert violation is not None
    assert violation["violation_type"] == "forbidden_target"
