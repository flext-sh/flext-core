from __future__ import annotations

import types
from pathlib import Path

from flext_core import r
from flext_infra.deps.internal_sync import FlextInfraInternalDependencySyncService
from flext_tests import tm
from tests.infra import h


def _set_toml_sequence(
    service: FlextInfraInternalDependencySyncService,
    values: list[object],
) -> None:
    state = {"index": 0}

    def _next(_path: Path) -> object:
        item = values[state["index"]]
        state["index"] += 1
        return item

    service.toml = types.SimpleNamespace(read_plain=_next)


class TestCollectInternalDepsEdgeCases:
    def test_collect_internal_deps_variants(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("x")

        def _collect(value: object):
            service = FlextInfraInternalDependencySyncService()
            _set_toml_sequence(service, [value])
            result = service.collect_internal_deps(tmp_path)
            tm.ok(result)
            return result

        one_result = _collect(
            r[dict[str, object]].ok({
                "tool": {
                    "poetry": {
                        "dependencies": {"flext-core": {"path": "../flext-core"}}
                    }
                },
                "project": {},
            })
        )
        two_result = _collect(
            r[dict[str, object]].ok({
                "tool": {},
                "project": {"dependencies": ["flext-core @ file:../flext-core"]},
            })
        )
        three_result = _collect(
            r[dict[str, object]].ok({
                "tool": {
                    "poetry": {
                        "dependencies": {"external-lib": {"path": "some/nested/path"}}
                    }
                },
                "project": {},
            })
        )
        four_result = _collect(
            r[dict[str, object]].ok({
                "tool": {"poetry": {"dependencies": {"flext-core": {"path": 123}}}},
                "project": {},
            })
        )
        five_result = _collect(
            r[dict[str, object]].ok({
                "tool": {},
                "project": {"dependencies": ["flext-core @"]},
            })
        )
        six_result = _collect(
            r[dict[str, object]].ok({
                "tool": {},
                "project": {"dependencies": ["flext-core @ file:///external/path"]},
            })
        )
        one = one_result.value
        two = two_result.value
        three = three_result.value
        four = four_result.value
        five = five_result.value
        six = six_result.value
        tm.that("flext-core" in one, eq=True)
        tm.that("flext-core" in two, eq=True)
        tm.that("external-lib" in three, eq=False)
        tm.that(len(four), eq=0)
        tm.that(len(five), eq=0)
        tm.that(len(six), eq=0)
        tm.that(h is not None, eq=True)
