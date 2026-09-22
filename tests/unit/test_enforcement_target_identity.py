"""Public census rules distinguish config and facade ownership from class names."""

from __future__ import annotations

import pytest

from flext_cli import c
from flext_core import FlextConfig, FlextSettings
from tests import m, u


class TestsFlextCoreEnforcementTargetIdentity:
    """Exercise classification through the public enforcement report."""

    @pytest.mark.parametrize("indirect", [False, True])
    def test_config_lineage_is_not_settings(self, indirect: bool) -> None:
        parent = type("FlextParentConfig", (FlextConfig,), {}) if indirect else FlextConfig
        target = type("FlextWorkerConfig", (parent,), {"__module__": "flext_core.synthetic"})

        assert not any(v.rule_id == "ENFORCE-042" for v in u.check(target).violations)

    def test_settings_lineage_remains_valid(self) -> None:
        target = type(
            "FlextWorkerSettings", (FlextSettings,), {"__module__": "flext_core.synthetic"}
        )

        assert not any(v.rule_id == "ENFORCE-042" for v in u.check(target).violations)

    @pytest.mark.parametrize("lookalike_config", [False, True])
    def test_raw_settings_and_config_name_impostor_remain_invalid(
        self, lookalike_config: bool
    ) -> None:
        parent = (
            type("FlextConfig", (m.BaseSettings,), {})
            if lookalike_config
            else m.BaseSettings
        )
        target = type(
            "FlextWorkerSettings", (parent,), {"__module__": "flext_core.synthetic"}
        )

        assert any(v.rule_id == "ENFORCE-042" for v in u.check(target).violations)

    @pytest.mark.parametrize("multiple_bases", [False, True])
    @pytest.mark.parametrize("facade_module", [False, True])
    def test_only_declared_facade_modules_require_alias_first(
        self, multiple_bases: bool, facade_module: bool
    ) -> None:
        package = c.__module__.split(".", 1)[0]
        service_module = f"{package}.services.worker"
        parent = type(
            f"{c.__name__}Worker", (), {"__module__": service_module}
        )
        peer = type("Peer", (), {"__module__": service_module})
        target = type(
            c.__name__,
            (parent, peer) if multiple_bases else (parent,),
            {"__module__": c.__module__ if facade_module else service_module},
        )

        violations = {
            v.rule_id
            for v in u.check(target).violations
            if v.rule_id in {"ENFORCE-047", "ENFORCE-049"}
        }

        assert bool(violations) is facade_module

    def test_declared_facade_with_alias_first_remains_valid(self) -> None:
        target = type(
            c.__name__,
            (c,),
            {"__module__": c.__module__},
        )

        assert not any(
            v.rule_id in {"ENFORCE-047", "ENFORCE-049"}
            for v in u.check(target).violations
        )
