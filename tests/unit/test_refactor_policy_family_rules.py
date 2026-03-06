from __future__ import annotations

from pathlib import Path

from flext_infra.refactor.rules.class_nesting import PreCheckGate


def _policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "src"
        / "flext_infra"
        / "refactor"
        / "rules"
        / "class-policy-v2.yml"
    )


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
        "current_file": "flext-core/src/flext_core/_runtime_metadata.py",
        "target_namespace": "FlextDispatcher",
    })
    assert not ok
    assert violation is not None
    assert violation["violation_type"] == "forbidden_target"
