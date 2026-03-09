"""Refactor/CST helper utilities for infrastructure code analysis.

Centralizes CST (Concrete Syntax Tree) helpers previously defined as
module-level functions in ``flext_infra.refactor.analysis``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import libcst as cst
from pydantic import TypeAdapter, ValidationError

from flext_core import r
from flext_infra._utilities.io import FlextInfraUtilitiesIo
from flext_infra._utilities.subprocess import FlextInfraCommandRunner
from flext_infra._utilities.yaml import FlextInfraUtilitiesYaml
from flext_infra.constants import FlextInfraConstants as c
from flext_infra.typings import FlextInfraTypes as t


class FlextInfraUtilitiesRefactor:
    """CST/refactor helpers for code analysis.

    Usage via namespace::

        from flext_infra import u

        name = u.Infra.Refactor.dotted_name(cst_expr)
    """

    @staticmethod
    def capture_output(cmd: Sequence[str]) -> r[str]:
        """Run *cmd* and return stripped stdout as ``r[str]``."""
        return FlextInfraCommandRunner().capture(cmd)

    @staticmethod
    def dotted_name(expr: cst.BaseExpression) -> str:
        """Extract dotted name string from a CST expression.

        Recursively traverses ``Name`` and ``Attribute`` nodes to build
        a fully qualified dotted name (e.g., ``"foo.bar.baz"``).

        Args:
            expr: CST expression node.

        Returns:
            Dotted name string, or empty string for unsupported nodes.

        """
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            root = FlextInfraUtilitiesRefactor.dotted_name(expr.value)
            if not root:
                return ""
            return f"{root}.{expr.attr.value}"
        return ""

    @staticmethod
    def root_name(expr: cst.BaseExpression) -> str:
        """Extract the root (leftmost) name from a CST expression.

        Handles ``Name``, ``Attribute``, and ``Call`` nodes to find the
        base identifier.

        Args:
            expr: CST expression node.

        Returns:
            Root name string, or empty string for unsupported nodes.

        """
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            return FlextInfraUtilitiesRefactor.root_name(expr.value)
        if isinstance(expr, cst.Call):
            return FlextInfraUtilitiesRefactor.root_name(expr.func)
        return ""

    @staticmethod
    def asname_to_local(asname: cst.AsName | None) -> str | None:
        """Extract the local alias name from a CST AsName node.

        Args:
            asname: CST AsName node, or None.

        Returns:
            Local alias string, or None if no alias or unsupported node.

        """
        if asname is None:
            return None
        if isinstance(asname.name, cst.Name):
            return asname.name.value
        return None

    @staticmethod
    def discover_project_roots(*, workspace_root: Path) -> list[Path]:
        """Discover project roots under a workspace."""
        roots: list[Path] = []

        def _looks_like_project(path: Path) -> bool:
            if not path.is_dir():
                return False
            if not (path / c.Infra.Files.MAKEFILE_FILENAME).exists():
                return False
            has_pyproject = (path / c.Infra.Files.PYPROJECT_FILENAME).exists()
            has_gomod = (path / c.Infra.Files.GO_MOD).exists()
            if not has_pyproject and (not has_gomod):
                return False
            return any(
                (path / dir_name).is_dir()
                for dir_name in c.Infra.Refactor.MRO_SCAN_DIRECTORIES
            )

        if _looks_like_project(workspace_root):
            roots.append(workspace_root)

        for entry in sorted(workspace_root.iterdir(), key=lambda item: item.name):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if not _looks_like_project(entry):
                continue
            roots.append(entry)
        if (
            len(roots) == 0
            and (workspace_root / c.Infra.Paths.DEFAULT_SRC_DIR).is_dir()
        ):
            roots = [workspace_root]
        return roots

    @staticmethod
    def iter_python_files(
        *,
        workspace_root: Path,
        include_tests: bool = True,
        include_examples: bool = True,
        include_scripts: bool = True,
    ) -> list[Path]:
        """Iterate Python files across all projects in a workspace.

        Args:
            workspace_root: Top-level workspace directory.
            include_tests: Whether to include files under ``tests/``.

        Returns:
            Sorted list of ``.py`` file paths.

        """
        roots = FlextInfraUtilitiesRefactor.discover_project_roots(
            workspace_root=workspace_root,
        )
        files: list[Path] = []
        for project_root in roots:
            src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
            if src_dir.is_dir():
                files.extend(sorted(src_dir.rglob(c.Infra.Extensions.PYTHON_GLOB)))
            if include_examples:
                examples_dir = project_root / c.Infra.Directories.EXAMPLES
                if examples_dir.is_dir():
                    files.extend(
                        sorted(examples_dir.rglob(c.Infra.Extensions.PYTHON_GLOB)),
                    )
            if include_scripts:
                scripts_dir = project_root / c.Infra.Directories.SCRIPTS
                if scripts_dir.is_dir():
                    files.extend(
                        sorted(scripts_dir.rglob(c.Infra.Extensions.PYTHON_GLOB)),
                    )
            if include_tests:
                tests_dir = project_root / c.Infra.Directories.TESTS
                if tests_dir.is_dir():
                    files.extend(
                        sorted(tests_dir.rglob(c.Infra.Extensions.PYTHON_GLOB)),
                    )
        return files

    @staticmethod
    def module_path(*, file_path: Path, project_root: Path) -> str:
        """Compute dotted module path relative to a project root.

        Strips the ``src/`` directory component and file extension.

        Args:
            file_path: Absolute path to a Python file.
            project_root: Root directory of the project.

        Returns:
            Dotted module path (e.g., ``"flext_infra.refactor.engine"``).

        """
        rel = file_path.relative_to(project_root)
        parts = [part for part in rel.with_suffix("").parts if part != "src"]
        return ".".join(parts)

    @staticmethod
    def module_family_from_path(path: str) -> str:
        """Resolve module family key from a source file path."""
        normalized = path.replace("\\", "/")
        if "_models" in normalized:
            return "_models"
        if "_utilities" in normalized:
            return "_utilities"
        if "_dispatcher" in normalized:
            return "_dispatcher"
        if "_decorators" in normalized:
            return "_decorators"
        if "_runtime" in normalized:
            return "_runtime"
        return "other_private"

    @staticmethod
    def entry_list(value: t.Infra.InfraValue | None) -> list[dict[str, str]]:
        """Normalize class-nesting config entries to a strict list."""
        if value is None:
            return []
        try:
            return TypeAdapter(list[dict[str, str]]).validate_python(value)
        except ValidationError:
            msg = "class nesting entries must be a list"
            raise ValueError(msg) from None

    @staticmethod
    def string_list(value: t.ContainerValue | None) -> list[str]:
        """Normalize policy fields that should contain string collections."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            items: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    msg = "expected list[str] value"
                    raise TypeError(msg)
                items.append(item)
            return items
        msg = "expected list[str] value"
        raise ValueError(msg)

    @staticmethod
    def mapping_list(
        value: t.ContainerValue | None,
    ) -> list[dict[str, t.ContainerValue]]:
        """Normalize policy fields that should contain mapping collections."""
        if value is None:
            return []
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        msg = "expected list[dict[str, ContainerValue]] value"
        raise ValueError(msg)

    @staticmethod
    def has_required_fields(
        entry: t.ContainerValue,
        required_fields: list[str],
    ) -> bool:
        if not isinstance(entry, dict):
            return False
        return all(key in entry for key in required_fields)

    @staticmethod
    def normalize_module_path(path_value: Path) -> str:
        normalized = path_value.as_posix().replace("\\", "/")
        path = Path(normalized)
        parts = path.parts
        if c.Infra.Paths.DEFAULT_SRC_DIR in parts:
            src_index = parts.index(c.Infra.Paths.DEFAULT_SRC_DIR)
            suffix = parts[src_index + 1 :]
            if suffix:
                return Path(*suffix).as_posix()
        return path.as_posix().lstrip("./")

    @staticmethod
    def project_scope_tokens(path_value: Path) -> set[str]:
        normalized = path_value.as_posix().replace("\\", "/")
        parts = Path(normalized).parts
        if not parts:
            return set()
        tokens: set[str] = set()
        if c.Infra.Paths.DEFAULT_SRC_DIR in parts:
            src_index = parts.index(c.Infra.Paths.DEFAULT_SRC_DIR)
            if src_index > 0:
                tokens.add(parts[src_index - 1])
            if src_index + 1 < len(parts):
                tokens.add(parts[src_index + 1])
        return tokens

    @staticmethod
    def rewrite_scope(entry: t.Infra.StrMap) -> str:
        raw_scope = entry.get(c.Infra.ReportKeys.REWRITE_SCOPE, c.Infra.ReportKeys.FILE)
        scope = raw_scope.strip().lower()
        if scope in {
            c.Infra.ReportKeys.FILE,
            c.Infra.Toml.PROJECT,
            c.Infra.ReportKeys.WORKSPACE,
        }:
            return scope
        msg = f"unsupported rewrite_scope: {raw_scope}"
        raise ValueError(msg)

    @staticmethod
    def scope_applies_to_file(
        entry: t.Infra.StrMap,
        current_file: Path,
        candidate_file: Path,
    ) -> bool:
        rewrite_scope = FlextInfraUtilitiesRefactor.rewrite_scope(entry)
        if rewrite_scope == c.Infra.ReportKeys.WORKSPACE:
            return True
        current_module = FlextInfraUtilitiesRefactor.normalize_module_path(current_file)
        candidate_module = FlextInfraUtilitiesRefactor.normalize_module_path(
            candidate_file
        )
        if rewrite_scope == c.Infra.ReportKeys.FILE:
            return current_module == candidate_module
        current_tokens = FlextInfraUtilitiesRefactor.project_scope_tokens(current_file)
        candidate_tokens = FlextInfraUtilitiesRefactor.project_scope_tokens(
            candidate_file
        )
        if current_tokens and candidate_tokens:
            return bool(current_tokens & candidate_tokens)
        return current_module == candidate_module

    @staticmethod
    def policy_document_schema_valid(
        loaded: dict[str, t.ContainerValue],
        schema_path: Path,
    ) -> bool:
        schema_result = FlextInfraUtilitiesIo.read_json(schema_path)
        if schema_result.is_failure:
            return False
        raw_schema: Mapping[str, t.ContainerValue] = schema_result.value
        schema: dict[str, t.ContainerValue] = dict(raw_schema.items())
        top_required = FlextInfraUtilitiesRefactor.string_list(
            schema.get("required", [])
        )
        if not FlextInfraUtilitiesRefactor.has_required_fields(loaded, top_required):
            return False
        definitions_raw = schema.get("definitions", {})
        if not isinstance(definitions_raw, dict):
            return False
        policy_entry_raw = definitions_raw.get("policyEntry", {})
        class_rule_raw = definitions_raw.get("classRule", {})
        if not isinstance(policy_entry_raw, dict):
            return False
        if not isinstance(class_rule_raw, dict):
            return False
        policy_entry_required = FlextInfraUtilitiesRefactor.string_list(
            policy_entry_raw.get("required", []),
        )
        class_rule_required = FlextInfraUtilitiesRefactor.string_list(
            class_rule_raw.get("required", []),
        )
        for entry in FlextInfraUtilitiesRefactor.mapping_list(
            loaded.get("policy_matrix")
        ):
            if not FlextInfraUtilitiesRefactor.has_required_fields(
                entry,
                policy_entry_required,
            ):
                return False
        for rule in FlextInfraUtilitiesRefactor.mapping_list(
            loaded.get(c.Infra.ReportKeys.RULES),
        ):
            if not FlextInfraUtilitiesRefactor.has_required_fields(
                rule, class_rule_required
            ):
                return False
        return True

    @staticmethod
    def load_validated_policy_document(policy_path: Path) -> t.Infra.ContainerDict:
        try:
            loaded = FlextInfraUtilitiesYaml.safe_load_yaml(policy_path)
        except (OSError, TypeError) as exc:
            msg = f"failed to read policy document: {policy_path}"
            raise ValueError(msg) from exc
        loaded_dict = dict(loaded.items())
        schema_path = policy_path.with_name("class-policy-v2.schema.json")
        if not FlextInfraUtilitiesRefactor.policy_document_schema_valid(
            loaded_dict,
            schema_path,
        ):
            msg = "policy document failed schema validation"
            raise ValueError(msg)
        return loaded_dict


__all__ = ["FlextInfraUtilitiesRefactor"]
