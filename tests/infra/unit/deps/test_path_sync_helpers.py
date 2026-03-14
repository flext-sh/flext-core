from __future__ import annotations

import pytest

from flext_core import r
from flext_infra.deps.path_sync import FlextInfraDependencyPathSync
from flext_tests import tm

_PATH_SYNC = FlextInfraDependencyPathSync()
_extract_requirement_name = FlextInfraDependencyPathSync._extract_requirement_name
_target_path = _PATH_SYNC._target_path
extract_dep_name = FlextInfraDependencyPathSync.extract_dep_name


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("flext-core", "flext-core"),
        (".flext-deps/flext-core", "flext-core"),
        ("../flext-core", "flext-core"),
        ("/flext-core", "flext-core"),
        ("  flext-core  ", "flext-core"),
        ("./flext-core", "flext-core"),
        ("/../flext-core", "flext-core"),
        ("", ""),
    ],
    ids=[
        "simple",
        "prefix",
        "parent-ref",
        "leading-slash",
        "whitespace",
        "dot-prefix",
        "parent-and-slash",
        "empty",
    ],
)
def test_extract_dep_name(source: str, expected: str) -> None:
    tm.that(extract_dep_name(source), eq=expected)


@pytest.mark.parametrize(
    ("dep_name", "is_root", "mode", "expected"),
    [
        ("flext-core", True, "workspace", "flext-core"),
        ("flext-core", False, "workspace", "../flext-core"),
        ("flext-core", True, "standalone", ".flext-deps/flext-core"),
        ("flext-core", False, "standalone", ".flext-deps/flext-core"),
    ],
    ids=[
        "workspace-root",
        "workspace-subproject",
        "standalone-root",
        "standalone-subproject",
    ],
)
def test_target_path(dep_name: str, is_root: bool, mode: str, expected: str) -> None:
    tm.that(_target_path(dep_name, is_root=is_root, mode=mode), eq=expected)


@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        ("flext-core @ file://.flext-deps/flext-core", "flext-core"),
        ("flext-core", "flext-core"),
        ("flext-core>=1.0.0", "flext-core"),
        ("@invalid", None),
        ("", None),
        (
            'flext-core @ file://.flext-deps/flext-core ; python_version >= "3.8"',
            "flext-core",
        ),
        ("flext-core @ file:../flext-core", "flext-core"),
        ("requests>=2.0", "requests"),
    ],
    ids=[
        "pep621-path",
        "simple-name",
        "versioned",
        "invalid",
        "empty",
        "with-marker",
        "path-dependency",
        "simple-package",
    ],
)
def test_extract_requirement_name(requirement: str, expected: str | None) -> None:
    tm.that(_extract_requirement_name(requirement), eq=expected)


def test_helpers_alias_is_reachable_helpers() -> None:
    tm.fail(r[bool].fail("x"), has="x")
