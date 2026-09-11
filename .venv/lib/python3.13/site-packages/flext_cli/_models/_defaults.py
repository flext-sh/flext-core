"""Immutable default values shared by CLI declaration models."""

from __future__ import annotations

from types import MappingProxyType

from flext_cli import t


EMPTY_JSON_MAPPING: t.JsonMapping = MappingProxyType({})
EMPTY_STR_MAPPING: t.StrMapping = MappingProxyType({})


__all__: tuple[str, ...] = ()
