"""Fixture DSL mixin for project-specific test classes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from flext_tests import t


class FlextTestsFixturesDSLMixin:
    """DSL mixin that exposes flat fixture-loading verbs on a project Tests class.

    Usage — in your project's tests/utilities.py::

        class Tests(FlextTestsFixturesDSLMixin):
            _FIXTURES_ROOT = c.Project.Tests.FIXTURES_DIR
            _FILE_EXTENSION = ".ldif"  # optional, default is ".ldif"

    Then test code calls::

        content = u.Project.Tests.load("oid", "schema")
        all_s = u.Project.Tests.for_kind("schema")  # {server: content}
        all_k = u.Project.Tests.for_group("oid")  # {kind: content}
        nested = u.Project.Tests.all_fixtures()  # {server: {kind: content}}
        servers = u.Project.Tests.servers()  # ("oid", "oud", …)
        kinds = u.Project.Tests.kinds("oid")  # ("acl", "entries", …)
        p_params = u.Project.Tests.pytest_params("schema")  # [(server, content), …]
    """

    _FIXTURES_ROOT: ClassVar[Path | None] = None
    _FILE_EXTENSION: ClassVar[str] = ".ldif"

    @classmethod
    def _root(cls) -> Path:
        if cls._FIXTURES_ROOT is None:
            msg = f"{cls.__name__} must set _FIXTURES_ROOT: ClassVar[Path]"
            raise TypeError(msg)
        return cls._FIXTURES_ROOT

    @classmethod
    def _resolve_path(cls, group: str, kind: str) -> Path:
        """Compute the candidate fixture path without checking existence."""
        root = cls._root()
        return root / group / f"{group}_{kind}_fixtures{cls._FILE_EXTENSION}"

    @classmethod
    def path(cls, group: str, kind: str) -> Path:
        fp = cls._resolve_path(group, kind)
        if not fp.exists():
            msg = f"Fixture not found: {fp}"
            raise FileNotFoundError(msg)
        return fp

    @classmethod
    def load(cls, group: str, kind: str) -> str:
        return cls.path(group, kind).read_text(encoding="utf-8")

    @classmethod
    def exists(cls, group: str, kind: str) -> bool:
        return cls._resolve_path(group, kind).exists()

    @classmethod
    def servers(cls) -> t.StrSequence:
        root = cls._root()
        if not root.exists():
            return ()
        return tuple(sorted(d.name for d in root.iterdir() if d.is_dir()))

    @classmethod
    def kinds(cls, group: str) -> t.StrSequence:
        root = cls._root()
        server_dir = root / group
        if not server_dir.exists():
            return ()
        prefix = f"{group}_"
        suffix = f"_fixtures{cls._FILE_EXTENSION}"
        return tuple(
            sorted(
                name[len(prefix) : -len(suffix)]
                for name in sorted(e.name for e in server_dir.iterdir())
                if name.startswith(prefix) and name.endswith(suffix)
            )
        )

    @classmethod
    def for_group(cls, group: str) -> t.MappingKV[str, str]:
        """All kinds for one server: {kind: content}."""
        return {kind: cls.load(group, kind) for kind in cls.kinds(group)}

    @classmethod
    def for_kind(cls, kind: str) -> t.MappingKV[str, str]:
        """All servers that have that kind: {server: content}."""
        return {
            server: cls.load(server, kind)
            for server in cls.servers()
            if cls.exists(server, kind)
        }

    @classmethod
    def all_fixtures(cls) -> t.MappingKV[str, t.MappingKV[str, str]]:
        """Full nested fixture dict: {server: {kind: content}}."""
        return {server: cls.for_group(server) for server in cls.servers()}

    @classmethod
    def pytest_params(cls, kind: str) -> t.StrPairSequence:
        """All (server, content) tuples for a given kind — ready for parametrize."""
        return [
            (server, cls.load(server, kind))
            for server in cls.servers()
            if cls.exists(server, kind)
        ]

    @classmethod
    def all_pytest_params(cls) -> t.SequenceOf[tuple[str, str, str]]:
        """All (server, kind, content) triples — ready for full-matrix parametrize."""
        return [
            (server, kind, cls.load(server, kind))
            for server in cls.servers()
            for kind in cls.kinds(server)
        ]
