from __future__ import annotations

import ast
from pathlib import Path

import libcst as cst
import pytest

from flext_infra import c, u


class TestParsingModuleAst:
    def test_parse_module_ast_returns_module_on_valid_python(
        self, tmp_path: Path
    ) -> None:
        file_path = tmp_path / "valid.py"
        file_path.write_text("value = 1\n", encoding=c.Infra.Encoding.DEFAULT)

        parsed = u.Infra.parse_module_ast(file_path)

        assert parsed is not None
        assert isinstance(parsed, ast.Module)

    def test_parse_module_ast_returns_none_on_syntax_error(
        self, tmp_path: Path
    ) -> None:
        file_path = tmp_path / "invalid.py"
        file_path.write_text(
            "def broken(:\n    pass\n", encoding=c.Infra.Encoding.DEFAULT
        )

        parsed = u.Infra.parse_module_ast(file_path)

        assert parsed is None

    def test_parse_module_ast_returns_none_for_nonexistent_file(
        self, tmp_path: Path
    ) -> None:
        parsed = u.Infra.parse_module_ast(tmp_path / "missing.py")

        assert parsed is None

    def test_parse_module_ast_returns_none_on_permission_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        file_path = tmp_path / "blocked.py"
        file_path.write_text("x = 1\n", encoding=c.Infra.Encoding.DEFAULT)

        def _raise_oserror(_self: Path, **_kwargs: str) -> str:
            msg = "permission denied"
            raise OSError(msg)

        monkeypatch.setattr(Path, "read_text", _raise_oserror)

        parsed = u.Infra.parse_module_ast(file_path)

        assert parsed is None


class TestParsingModuleCst:
    def test_parse_module_cst_returns_module_on_valid_python(
        self, tmp_path: Path
    ) -> None:
        file_path = tmp_path / "valid.py"
        file_path.write_text("value = 1\n", encoding=c.Infra.Encoding.DEFAULT)

        parsed = u.Infra.parse_module_cst(file_path)

        assert parsed is not None
        assert isinstance(parsed, cst.Module)

    def test_parse_module_cst_returns_none_on_syntax_error(
        self, tmp_path: Path
    ) -> None:
        file_path = tmp_path / "invalid.py"
        file_path.write_text(
            "def broken(:\n    pass\n", encoding=c.Infra.Encoding.DEFAULT
        )

        parsed = u.Infra.parse_module_cst(file_path)

        assert parsed is None

    def test_parse_module_cst_returns_none_for_nonexistent_file(
        self, tmp_path: Path
    ) -> None:
        parsed = u.Infra.parse_module_cst(tmp_path / "missing.py")

        assert parsed is None

    def test_parse_module_cst_returns_none_on_permission_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        file_path = tmp_path / "blocked.py"
        file_path.write_text("x = 1\n", encoding=c.Infra.Encoding.DEFAULT)

        def _raise_oserror(_self: Path, **_kwargs: str) -> str:
            msg = "permission denied"
            raise OSError(msg)

        monkeypatch.setattr(Path, "read_text", _raise_oserror)

        parsed = u.Infra.parse_module_cst(file_path)

        assert parsed is None
