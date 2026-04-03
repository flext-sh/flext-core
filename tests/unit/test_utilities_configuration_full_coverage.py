"""Tests for Configuration utilities full coverage."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from tests import p, r, t, u
from tests.unit import _models_impl as test_unit_models


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
        assert u.Infra.resolve_env_file() == str(existing.resolve())
        missing = tmp_path / "missing.env"
        monkeypatch.setenv("FLEXT_ENV_FILE", str(missing))
        assert u.Infra.resolve_env_file() == str(missing)
        monkeypatch.delenv("FLEXT_ENV_FILE", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("B=2\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert u.Infra.resolve_env_file() == str(env_file.resolve())
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
                cast("t.NormalizedValue", test_unit_models._DumpErrorModel()),
            ),
            "missing",
        ) == (False, None)
        assert u._try_get_from_duck_model_dump(self._DuckDumpError(), "value") == (
            False,
            None,
        )
