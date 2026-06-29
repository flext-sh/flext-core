"""Beartype config and facade tests."""

from __future__ import annotations

import pytest
from beartype import BeartypeConf, BeartypeStrategy

from flext_core._utilities.beartype_conf import FlextUtilitiesBeartypeConf
from tests import c, u
from tests.unit._beartype_engine_support import (
    TestsFlextBeartypeEngine,
)


class TestsFlextBeartypeEngineConfig(TestsFlextBeartypeEngine):
    def test_default_mode_conf(self) -> None:
        """Default beartype mode is disabled in flext_core."""
        conf = FlextUtilitiesBeartypeConf.build_beartype_conf()
        assert conf.strategy is BeartypeStrategy.O0

    def test_default_mode_strategy(self) -> None:
        """Disabled default mode uses O0 strategy."""
        conf = FlextUtilitiesBeartypeConf.build_beartype_conf()
        assert conf.strategy is BeartypeStrategy.O0

    def test_conf_is_beartype_conf(self) -> None:
        """Factory returns a proper BeartypeConf instance."""
        conf = FlextUtilitiesBeartypeConf.build_beartype_conf()
        assert isinstance(conf, BeartypeConf)

    def test_beartype_mode_matches_default(self) -> None:
        """flext_core starts with beartype activation disabled by default."""
        assert c.BEARTYPE_MODE is c.EnforcementMode.OFF

    @pytest.mark.parametrize(
        "method",
        [
            "contains_any",
            "has_forbidden_collection_origin",
            "count_union_members",
            "matches_str_none_union",
            "alias_contains_any",
            "build_beartype_conf",
        ],
    )
    def test_all_methods_on_facade(self, method: str) -> None:
        """All beartype engine + conf methods on u.*."""
        assert hasattr(u, method) is True
