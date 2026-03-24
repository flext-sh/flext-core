"""Tests for Configuration utilities full coverage."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from tests import p, r, t, u

from ._models import TestUnitModels


class TestUtilitiesConfigurationFullCoverage:
    class _DuckDumpError:
        model_dump = "duck boom"

    class _ContainerOK:
        def register(
            self,
            _name: str,
            _instance: t.NormalizedValue,
            **_kwargs: t.Scalar,
        ) -> r[bool]:
            return r[bool].ok(True)

        def register_factory(
            self,
            _name: str,
            _factory: Callable[[], t.NormalizedValue],
        ) -> r[bool]:
            return r[bool].ok(True)

    class _ContainerFail:
        def register(
            self,
            _name: str,
            _instance: t.NormalizedValue,
            **_kwargs: t.Scalar,
        ) -> r[bool]:
            return r[bool].fail("reg fail")

        def register_factory(
            self,
            _name: str,
            _factory: Callable[[], t.NormalizedValue],
        ) -> r[bool]:
            return r[bool].fail("fac fail")

    class _ContainerRaise:
        def register(
            self,
            _name: str,
            _instance: t.NormalizedValue,
            **_kwargs: t.Scalar,
        ) -> r[bool]:
            msg = "reg ex"
            raise RuntimeError(msg)

        def register_factory(
            self,
            _name: str,
            _factory: Callable[[], t.NormalizedValue],
        ) -> r[bool]:
            msg = "fac ex"
            raise RuntimeError(msg)

    def test_resolve_env_file_and_log_level(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        existing = tmp_path / "custom.env"
        existing.write_text("A=1\n", encoding="utf-8")
        monkeypatch.setenv("FLEXT_ENV_FILE", str(existing))
        assert u.resolve_env_file() == str(existing.resolve())
        missing = tmp_path / "missing.env"
        monkeypatch.setenv("FLEXT_ENV_FILE", str(missing))
        assert u.resolve_env_file() == str(missing)
        monkeypatch.delenv("FLEXT_ENV_FILE", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("B=2\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert u.resolve_env_file() == str(env_file.resolve())
        assert isinstance(u.get_log_level_from_config(), int)
        assert u.get_log_level_from_config() in {
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        }

    def test_private_getters_exception_paths(self) -> None:
        assert u._try_get_from_model_dump(
            cast(
                "p.HasModelDump",
                cast("t.NormalizedValue", TestUnitModels._DumpErrorModel()),
            ),
            "missing",
        ) == (False, None)
        assert u._try_get_from_duck_model_dump(self._DuckDumpError(), "value") == (
            False,
            None,
        )

    def test_build_options_invalid_only_kwargs_returns_base(self) -> None:
        base = TestUnitModels._Opts(value=9)
        result = u.build_options_from_kwargs(
            model_class=TestUnitModels._Opts,
            explicit_options=base,
            default_factory=TestUnitModels._Opts,
            invalid_field=10,
        )
        assert result.is_success
        assert result.value is base

    def test_register_singleton_register_factory_and_bulk_register_paths(
        self,
    ) -> None:
        ok = cast("p.Container", cast("t.NormalizedValue", self._ContainerOK()))
        fail = cast("p.Container", cast("t.NormalizedValue", self._ContainerFail()))
        err = cast("p.Container", cast("t.NormalizedValue", self._ContainerRaise()))
        singleton_ok = u.register_singleton(ok, "s", 1)
        singleton_fail = u.register_singleton(fail, "s", 1)
        singleton_err = u.register_singleton(err, "s", 1)
        assert singleton_ok.is_success
        assert singleton_fail.is_failure
        assert singleton_err.is_failure
        factory_ok = u.register_factory(ok, "f", lambda: 1, _cache=True)
        factory_fail = u.register_factory(fail, "f", lambda: 1)
        factory_err = u.register_factory(err, "f", lambda: 1)
        assert factory_ok.is_success
        assert factory_fail.is_failure
        assert factory_err.is_failure
        bulk_ok = u.bulk_register(ok, {"a": 1, "b": 2})
        assert bulk_ok.is_success
        assert bulk_ok.value == 2
        bulk_fail = u.bulk_register(fail, {"a": 1})
        assert bulk_fail.is_failure
        bulk_err = u.bulk_register(err, {"a": 1})
        assert bulk_err.is_failure
