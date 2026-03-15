from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from flext_core import r
from flext_infra import FlextInfraInternalDependencySyncService
from flext_tests import tm
from tests.infra import t


class _TomlReaderStub:
    """Test stub satisfying p.Infra.TomlReader protocol."""

    def __init__(
        self,
        fn: Callable[[Path], r[t.Infra.TomlConfig]],
    ) -> None:
        self._fn = fn

    def read_plain(self, path: Path) -> r[t.Infra.TomlConfig]:
        """Delegate to the callable provided at construction."""
        return self._fn(path)


def _set_toml_stub(
    service: FlextInfraInternalDependencySyncService,
    value: r[t.Infra.TomlConfig],
) -> None:
    def _reader(_path: Path) -> r[t.Infra.TomlConfig]:
        return value

    service.toml = _TomlReaderStub(fn=_reader)


def _set_toml_sequence(
    service: FlextInfraInternalDependencySyncService,
    values: list[r[t.Infra.TomlConfig]],
) -> None:
    state = {"index": 0}

    def _next(_path: Path) -> r[t.Infra.TomlConfig]:
        item = values[state["index"]]
        state["index"] += 1
        return item

    service.toml = _TomlReaderStub(fn=_next)


class TestParseGitmodules:
    def test_parse_gitmodules_valid(self, tmp_path: Path) -> None:
        gitmodules = tmp_path / ".gitmodules"
        gitmodules.write_text(
            '[submodule "flext-core"]\n\tpath = flext-core\n\turl = git@github.com:flext-sh/flext-core.git\n[submodule "flext-api"]\n\tpath = flext-api\n\turl = git@github.com:flext-sh/flext-api.git\n',
        )
        result = FlextInfraInternalDependencySyncService().parse_gitmodules(gitmodules)
        tm.that("flext-core" in result, eq=True)
        tm.that("flext-api" in result, eq=True)
        tm.that(
            result["flext-core"].ssh_url, eq="git@github.com:flext-sh/flext-core.git"
        )
        tm.that(result["flext-core"].https_url.startswith("https://"), eq=True)

    def test_parse_gitmodules_empty(self, tmp_path: Path) -> None:
        path = tmp_path / ".gitmodules"
        path.write_text("")
        tm.that(FlextInfraInternalDependencySyncService().parse_gitmodules(path), eq={})

    def test_parse_gitmodules_no_url(self, tmp_path: Path) -> None:
        path = tmp_path / ".gitmodules"
        path.write_text('[submodule "test"]\n\tpath = test\n')
        tm.that(FlextInfraInternalDependencySyncService().parse_gitmodules(path), eq={})

    def test_parse_gitmodules_non_submodule_section(self, tmp_path: Path) -> None:
        path = tmp_path / ".gitmodules"
        path.write_text("[other]\nfoo = bar\n")
        tm.that(FlextInfraInternalDependencySyncService().parse_gitmodules(path), eq={})


class TestParseRepoMap:
    def test_parse_repo_map_success(self) -> None:
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            r[t.Infra.TomlConfig].ok({
                "repo": {
                    "flext-core": {
                        "ssh_url": "git@github.com:flext-sh/flext-core.git",
                        "https_url": "https://github.com/flext-sh/flext-core.git",
                    }
                }
            }),
        )
        result = service.parse_repo_map(Path("/fake/map.toml"))
        tm.ok(result)
        tm.that("flext-core" in result.value, eq=True)

    def test_parse_repo_map_read_failure(self) -> None:
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(service, r[t.Infra.TomlConfig].fail("file not found"))
        tm.fail(service.parse_repo_map(Path("/fake/map.toml")))

    def test_parse_repo_map_no_repo_section(self) -> None:
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(service, r[t.Infra.TomlConfig].ok({"other": "data"}))
        tm.ok(service.parse_repo_map(Path("/fake/map.toml")), eq={})

    def test_parse_repo_map_non_dict_repo(self) -> None:
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(service, r[t.Infra.TomlConfig].ok({"repo": "not-a-dict"}))
        tm.ok(service.parse_repo_map(Path("/fake/map.toml")), eq={})

    def test_parse_repo_map_non_dict_values(self) -> None:
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            r[t.Infra.TomlConfig].ok({"repo": {"flext-core": "string-value"}}),
        )
        tm.ok(service.parse_repo_map(Path("/fake/map.toml")), eq={})

    def test_parse_repo_map_no_ssh_url(self) -> None:
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            r[t.Infra.TomlConfig].ok({"repo": {"flext-core": {"other": "val"}}}),
        )
        tm.ok(service.parse_repo_map(Path("/fake/map.toml")), eq={})

    def test_parse_repo_map_auto_https(self) -> None:
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            r[t.Infra.TomlConfig].ok({
                "repo": {
                    "flext-core": {"ssh_url": "git@github.com:flext-sh/flext-core.git"}
                }
            }),
        )
        result = service.parse_repo_map(Path("/fake/map.toml"))
        tm.ok(result)
        tm.that(result.value["flext-core"].https_url.startswith("https://"), eq=True)


class TestCollectInternalDeps:
    def test_no_pyproject(self, tmp_path: Path) -> None:
        tm.ok(
            FlextInfraInternalDependencySyncService().collect_internal_deps(tmp_path),
            eq={},
        )

    def test_poetry_path_deps(self, tmp_path: Path) -> None:
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            r[t.Infra.TomlConfig].ok({
                "tool": {
                    "poetry": {
                        "dependencies": {
                            "flext-core": {"path": ".flext-deps/flext-core"},
                            "requests": "^2.28",
                        }
                    }
                },
                "project": {},
            }),
        )
        (tmp_path / "pyproject.toml").write_text("")
        result = service.collect_internal_deps(tmp_path)
        tm.ok(result)
        tm.that("flext-core" in result.value, eq=True)

    def test_pep621_path_deps(self, tmp_path: Path) -> None:
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(
            service,
            r[t.Infra.TomlConfig].ok({
                "tool": {},
                "project": {
                    "dependencies": [
                        "flext-core @ file:.flext-deps/flext-core",
                        "requests>=2.28",
                    ]
                },
            }),
        )
        (tmp_path / "pyproject.toml").write_text("")
        result = service.collect_internal_deps(tmp_path)
        tm.ok(result)
        tm.that("flext-core" in result.value, eq=True)

    def test_read_failure(self, tmp_path: Path) -> None:
        service = FlextInfraInternalDependencySyncService()
        _set_toml_stub(service, r[t.Infra.TomlConfig].fail("parse error"))
        (tmp_path / "pyproject.toml").write_text("")
        tm.fail(service.collect_internal_deps(tmp_path))

    def test_no_tool_and_non_dict_deps(self, tmp_path: Path) -> None:
        service = FlextInfraInternalDependencySyncService()
        _set_toml_sequence(
            service,
            [
                r[t.Infra.TomlConfig].ok({"project": {}}),
                r[t.Infra.TomlConfig].ok({
                    "tool": {"poetry": {"dependencies": "not-a-dict"}},
                    "project": {"dependencies": "not-a-list"},
                }),
            ],
        )
        (tmp_path / "pyproject.toml").write_text("")
        tm.ok(service.collect_internal_deps(tmp_path), eq={})
        tm.ok(service.collect_internal_deps(tmp_path), eq={})
