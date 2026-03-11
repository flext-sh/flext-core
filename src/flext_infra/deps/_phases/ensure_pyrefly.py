"""Phase: Ensure standard Pyrefly configuration for max-strict typing."""

from __future__ import annotations

import tomlkit
from tomlkit.items import Table

from flext_infra import c
from flext_infra._utilities.toml import FlextInfraUtilitiesToml as _Toml
from flext_infra.deps.tool_config import FlextInfraToolConfigDocument

array = _Toml.array
as_string_list = _Toml.as_string_list
ensure_table = _Toml.ensure_table
toml_get = _Toml.get
unwrap_item = _Toml.unwrap_item


class EnsurePyreflyConfigPhase:
    """Ensure standard Pyrefly configuration for max-strict typing."""

    def __init__(self, tool_config: FlextInfraToolConfigDocument) -> None:
        self._tool_config = tool_config

    def apply(self, doc: tomlkit.TOMLDocument, *, is_root: bool) -> list[str]:
        changes: list[str] = []
        tool: object | None = None
        if c.Infra.Toml.TOOL in doc:
            tool = doc[c.Infra.Toml.TOOL]
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        pyrefly = ensure_table(tool, c.Infra.Toml.PYREFLY)
        if unwrap_item(toml_get(pyrefly, c.Infra.Toml.PYTHON_VERSION_HYPHEN)) != "3.13":
            pyrefly[c.Infra.Toml.PYTHON_VERSION_HYPHEN] = "3.13"
            changes.append("tool.pyrefly.python-version set to 3.13")
        if (
            unwrap_item(toml_get(pyrefly, c.Infra.Toml.IGNORE_ERRORS_IN_GENERATED))
            is not True
        ):
            pyrefly[c.Infra.Toml.IGNORE_ERRORS_IN_GENERATED] = True
            changes.append("tool.pyrefly.ignore-errors-in-generated-code enabled")
        expected_search = ["."]
        current_search = as_string_list(toml_get(pyrefly, c.Infra.Toml.SEARCH_PATH))
        if current_search != expected_search:
            pyrefly[c.Infra.Toml.SEARCH_PATH] = array(expected_search)
            changes.append(f"tool.pyrefly.search-path set to {expected_search}")
        errors = ensure_table(pyrefly, "errors")
        for error_rule in self._tool_config.tools.pyrefly.strict_errors:
            if unwrap_item(toml_get(errors, error_rule)) is not True:
                errors[error_rule] = True
                changes.append(f"tool.pyrefly.errors.{error_rule} enabled")
        for error_rule in self._tool_config.tools.pyrefly.disabled_errors:
            if unwrap_item(toml_get(errors, error_rule)) is not False:
                errors[error_rule] = False
                changes.append(f"tool.pyrefly.errors.{error_rule} disabled")
        current_excludes = as_string_list(
            toml_get(pyrefly, c.Infra.Toml.PROJECT_EXCLUDES),
        )
        pb2_globs = ["**/*_pb2*.py", "**/*_pb2_grpc*.py"]
        needed = set(pb2_globs) - set(current_excludes)
        if needed and (
            is_root or any(glob in current_excludes for glob in pb2_globs) or True
        ):
            pyrefly[c.Infra.Toml.PROJECT_EXCLUDES] = array(
                sorted(set(current_excludes) | set(pb2_globs)),
            )
            changes.append(f"tool.pyrefly.project-excludes added {', '.join(needed)}")
        return changes
