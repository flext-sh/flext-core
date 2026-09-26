"""Behavioral tests for the ENFORCE-046 concrete-namespace import predicate.

Exercises ``FlextUtilitiesBeartypeEngine.apply`` for the ``IMPORT_BLACKLIST``
kind on real on-disk canonical facade modules: a family facade that extends a
peer family facade by its class name (Pattern B) passes, while the same bare
class import outside a base, or in a package outside the family, is reported.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

from flext_core import c
from flext_core.models import FlextModelsEnforcement as me
from flext_core.utilities import FlextUtilitiesBeartypeEngine as be
from tests.typings import t


class TestsFlextCoreBeartypeEngineImportBlacklist:
    """Public-contract tests for the concrete-namespace import predicate."""

    @staticmethod
    def _facade(tmp_path: Path, package: str, body: str) -> type:
        """Materialize ``<package>/constants.py`` and import its ``Facade`` class."""
        root = tmp_path / package
        root.mkdir()
        (root / "__init__.py").write_text("", encoding="utf-8")
        (root / "constants.py").write_text(textwrap.dedent(body), encoding="utf-8")
        sys.path.insert(0, str(tmp_path))
        try:
            module = importlib.import_module(f"{package}.constants")
        finally:
            sys.path.remove(str(tmp_path))
        facade: type = module.Facade
        return facade

    @staticmethod
    def _apply(target: type) -> t.StrMapping | None:
        return be.apply(
            c.EnforcementPredicateKind.IMPORT_BLACKLIST,
            me.ImportBlacklistParams(),
            target,
        )

    def test_family_facade_extending_a_peer_by_class_name_passes(
        self, tmp_path: Path
    ) -> None:
        facade = self._facade(
            tmp_path,
            f"{c.NAMESPACE_FAMILY_PREFIX}peerprobe",
            """
            from flext_core import FlextConstants


            class Facade(FlextConstants):
                pass
            """,
        )
        assert self._apply(facade) is None

    def test_family_facade_holding_an_unused_bare_class_is_reported(
        self, tmp_path: Path
    ) -> None:
        facade = self._facade(
            tmp_path,
            f"{c.NAMESPACE_FAMILY_PREFIX}unusedprobe",
            """
            from flext_core import FlextConstants, FlextModels


            class Facade(FlextConstants):
                peer = FlextModels
            """,
        )
        assert self._apply(facade) == {
            "file": "constants.py",
            "import": "FlextModels",
        }

    def test_consumer_outside_the_family_keeps_the_alias_base(
        self, tmp_path: Path
    ) -> None:
        facade = self._facade(
            tmp_path,
            "consumerprobe",
            """
            from flext_core import FlextConstants


            class Facade(FlextConstants):
                pass
            """,
        )
        assert self._apply(facade) == {
            "file": "constants.py",
            "import": "FlextConstants",
        }
