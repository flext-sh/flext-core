"""Tests for runtime enforcement — query ``u.check(target)`` reports.

Every legacy per-rule helper (``u.check_no_any``, ``u.is_exempt``, etc.)
is gone. Tests assert over ``m.Report.violations`` filtered
by ``layer`` / ``severity`` / message fragment.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Generator
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from tests import m, p, t

_INVALID_ANNOTATION_SOURCE = """\
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any

from flext_core import m


class InvalidTypes:
    type BareAnyAlias = Any
    type AnyAlias = str | Any
    type NestedAnyAlias = Mapping[str, Any]
    type NestedAnySequenceAlias = Mapping[str, Sequence[Any]]
    type OptionalAnyAlias = Any | None
    type ObjectAlias = object
    type NestedObjectAlias = list[object]
    type CleanAlias = str | int


def shadowed_guard_owner():
    class Host:
        TYPE_CHECKING = False
        if TYPE_CHECKING:
            from pathlib import PurePath as GuardedPath
        type LazyAlias = GuardedPath | str
    return Host


def model_fixture():
    class BadModel(m.ArbitraryTypesModel):
        data: Annotated[Any, m.Field(description="Invalid annotation fixture")] = None
    return BadModel


BadModel = model_fixture()
"""


@contextmanager
def invalid_annotations(tmp_path: Path) -> Generator[ModuleType]:
    """Load deliberate invalid source while the native checker inspects it.

    Yields:
        The loaded module containing invalid annotation fixtures.

    Raises:
        ImportError: If the fixture has no native module loader.

    """
    path = tmp_path / "invalid_annotations_fixture.py"
    path.write_text(_INVALID_ANNOTATION_SOURCE, encoding="utf-8")
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = "Invalid annotation fixture has no native module loader"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def messages(report: m.Report, *, fragment: str) -> list[str]:
    return [v.message for v in report.violations if fragment in v.message]


def synthetic_method(_self: p.AttributeProbe) -> None:
    """Typed method body for synthetic classes built by :func:`make_class`."""
    return


def make_class[ValueT](
    name: str,
    body: t.MappingKV[str, ValueT],
) -> type:
    cls = type(name, (), dict(body))
    cls.__qualname__ = name  # strip test-method qualname prefix
    cls.__module__ = "flext_core.synthetic"
    return cls
