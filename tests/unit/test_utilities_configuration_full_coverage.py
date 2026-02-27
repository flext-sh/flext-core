"""Tests for Configuration utilities full coverage."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from flext_core import p, r, u
from flext_core.typings import JsonValue
from pydantic import BaseModel


class _DumpErrorModel(BaseModel):
    value: int = 1


class _DuckDumpError:
    model_dump = "duck boom"


class _Opts(BaseModel):
    value: int = 1


class _ContainerOK:
    def register(self, _name: str, _instance: JsonValue):
        return r[bool].ok(True)

    def register_factory(self, _name: str, _factory: Callable[[], object]) -> r[bool]:
        return r[bool].ok(True)


class _ContainerFail:
    def register(self, _name: str, _instance: JsonValue):
        return r[bool].fail("reg fail")

    def register_factory(self, _name: str, _factory: Callable[[], object]) -> r[bool]:
        return r[bool].fail("fac fail")


class _ContainerRaise:
    def register(self, _name: str, _instance: JsonValue):
        msg = "reg ex"
        raise RuntimeError(msg)

    def register_factory(self, _name: str, _factory: Callable[[], object]) -> r[bool]:
        msg = "fac ex"
        raise RuntimeError(msg)


def test_resolve_env_file_and_log_level(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing = tmp_path / "custom.env"
    existing.write_text("A=1\n", encoding="utf-8")
    monkeypatch.setenv("FLEXT_ENV_FILE", str(existing))
    assert u.Configuration.resolve_env_file() == str(existing.resolve())

    missing = tmp_path / "missing.env"
    monkeypatch.setenv("FLEXT_ENV_FILE", str(missing))
    assert u.Configuration.resolve_env_file() == str(missing)

    monkeypatch.delenv("FLEXT_ENV_FILE", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("B=2\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert u.Configuration.resolve_env_file() == str(env_file.resolve())

    assert isinstance(u.Configuration.get_log_level_from_config(), int)
    assert u.Configuration.get_log_level_from_config() in {
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    }


def test_private_getters_exception_paths() -> None:
    assert u.Configuration._try_get_from_model_dump(
        cast("p.HasModelDump", cast("object", _DumpErrorModel())),
        "missing",
    ) == (False, None)
    assert u.Configuration._try_get_from_duck_model_dump(_DuckDumpError(), "value") == (
        False,
        None,
    )


def test_build_options_invalid_only_kwargs_returns_base() -> None:
    base = _Opts(value=9)
    result = u.Configuration.build_options_from_kwargs(
        model_class=_Opts,
        explicit_options=base,
        default_factory=_Opts,
        invalid_field=10,
    )
    assert result.is_success
    assert result.value is base


def test_register_singleton_register_factory_and_bulk_register_paths() -> None:
    ok = cast("p.DI", cast("object", _ContainerOK()))
    fail = cast("p.DI", cast("object", _ContainerFail()))
    err = cast("p.DI", cast("object", _ContainerRaise()))

    singleton_ok = u.Configuration.register_singleton(ok, "s", 1)
    singleton_fail = u.Configuration.register_singleton(fail, "s", 1)
    singleton_err = u.Configuration.register_singleton(err, "s", 1)
    assert singleton_ok.is_success
    assert singleton_fail.is_failure
    assert singleton_err.is_failure

    factory_ok = u.Configuration.register_factory(ok, "f", lambda: 1, _cache=True)
    factory_fail = u.Configuration.register_factory(fail, "f", lambda: 1)
    factory_err = u.Configuration.register_factory(err, "f", lambda: 1)
    assert factory_ok.is_success
    assert factory_fail.is_failure
    assert factory_err.is_failure

    bulk_ok = u.Configuration.bulk_register(ok, {"a": 1, "b": 2})
    assert bulk_ok.is_success
    assert bulk_ok.value == 2

    bulk_fail = u.Configuration.bulk_register(fail, {"a": 1})
    assert bulk_fail.is_failure

    bulk_err = u.Configuration.bulk_register(err, {"a": 1})
    assert bulk_err.is_failure
