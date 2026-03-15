from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from flext_core import r
from flext_infra import FlextInfraInternalDependencySyncService
from flext_tests import tm
from tests.infra import h, t


def _set_toml_sequence(
    service: FlextInfraInternalDependencySyncService,
    values: list[r[t.Infra.TomlConfig]],
) -> None:
    state = {"index": 0}

    def _next(_path: Path) -> r[t.Infra.TomlConfig]:
        item = values[state["index"]]
        state["index"] += 1
        return item

    class _TomlReaderStub:
        def __init__(self, fn: Callable[[Path], r[t.Infra.TomlConfig]]) -> None:
            self._fn = fn

        def read_plain(self, path: Path) -> r[t.Infra.TomlConfig]:
            return self._fn(path)

    service.toml = _TomlReaderStub(_next)


class TestCollectInternalDepsEdgeCases:
    def test_collect_internal_deps_variants(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("x")

        def _collect(value: r[t.Infra.TomlConfig]) -> r[Mapping[str, Path]]:
            service = FlextInfraInternalDependencySyncService()
            _set_toml_sequence(service, [value])
            result = service.collect_internal_deps(tmp_path)
            tm.ok(result)
            return result

        one_result = _collect(
            r[t.Infra.TomlConfig].ok({
                "tool": {
                    "poetry": {
                        "dependencies": {"flext-core": {"path": "../flext-core"}}
                    }
                },
                "project": {},
            })
        )
        two_result = _collect(
            r[t.Infra.TomlConfig].ok({
                "tool": {},
                "project": {"dependencies": ["flext-core @ file:../flext-core"]},
            })
        )
        three_result = _collect(
            r[t.Infra.TomlConfig].ok({
                "tool": {
                    "poetry": {
                        "dependencies": {"external-lib": {"path": "some/nested/path"}}
                    }
                },
                "project": {},
            })
        )
        four_result = _collect(
            r[t.Infra.TomlConfig].ok({
                "tool": {"poetry": {"dependencies": {"flext-core": {"path": 123}}}},
                "project": {},
            })
        )
        five_result = _collect(
            r[t.Infra.TomlConfig].ok({
                "tool": {},
                "project": {"dependencies": ["flext-core @"]},
            })
        )
        six_result = _collect(
            r[t.Infra.TomlConfig].ok({
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
        tm.that(hasattr(h, "assert_ok"), eq=True)
