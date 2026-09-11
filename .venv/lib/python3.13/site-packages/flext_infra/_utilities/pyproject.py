"""Pyproject.toml parsing helpers for flext-infra utilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from functools import cache, lru_cache
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from flext_cli import u
from flext_core import r
from flext_infra import c, t
from flext_infra._utilities.git import FlextInfraUtilitiesGit

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraUtilitiesPyproject:
    """Static helpers for reading and normalizing ``pyproject.toml`` payloads."""

    @staticmethod
    def validate_infra_payload(payload: object) -> t.JsonMapping | None:
        """Validate one plain mapping through the infra adapter.

        Centralizes the repeated try/except so callers only decide what sentinel
        to surface on failure.
        """
        try:
            result: t.JsonMapping | None = (
                t.Infra.INFRA_MAPPING_ADAPTER.validate_python(payload)
            )
        except (c.ValidationError, ValueError):
            return None
        return result

    @classmethod
    def format_toml_source(
        cls, source: str, *, path: Path, toolchain_root: Path, taplo_version: str
    ) -> p.Result[str]:
        """Format TOML through the configured workspace Taplo toolchain."""
        config_path = toolchain_root / c.Infra.TAPLO_CONFIG_FILENAME
        config_content = config_path.read_bytes() if config_path.is_file() else b""
        resolved_path = path.resolve()
        resolved_toolchain_root = toolchain_root.resolve()
        execution_root = next(
            candidate
            for candidate in (resolved_toolchain_root, *resolved_toolchain_root.parents)
            if candidate.is_dir()
        )
        relative_path = (
            resolved_path.relative_to(resolved_toolchain_root).as_posix()
            if resolved_path.is_relative_to(resolved_toolchain_root)
            else resolved_path.relative_to(resolved_path.parent).as_posix()
        )
        return cls._format_toml_source_cached(
            source,
            relative_path=relative_path,
            config_path=config_path.resolve() if config_content else None,
            config_digest=sha256(config_content).hexdigest(),
            execution_root=execution_root,
            taplo_version=taplo_version,
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def _format_toml_source_cached(
        source: str,
        *,
        relative_path: str,
        config_path: Path | None,
        config_digest: str,
        execution_root: Path,
        taplo_version: str,
    ) -> p.Result[str]:
        del config_digest
        command = [
            "mise",
            "exec",
            f"taplo@{taplo_version}",
            "--",
            "taplo",
            "format",
            "-",
            "--stdin-filepath",
            relative_path,
        ]
        if config_path is not None:
            command.extend(("--config", str(config_path)))
        result = u.Cli.run_raw(
            command,
            cwd=execution_root,
            input_data=source.encode(c.Cli.ENCODING_DEFAULT),
        )
        if result.failure:
            return r[str].fail(result.error or "taplo format failed")
        output = result.value
        if output.exit_code != 0:
            detail = (output.stderr or output.stdout).strip()
            return r[str].fail(f"taplo format failed ({output.exit_code}): {detail}")
        return r[str].ok(output.stdout)

    @staticmethod
    @cache
    def pyproject_payload(pyproject_path: Path) -> t.JsonMapping:
        """Return one parsed ``pyproject.toml`` payload validated against ``t.Infra``.

        Disk read is delegated to ``u.Cli.toml_read_json`` (cached at
        flext-cli utility layer); this method caches the validated typed payload.
        ``pyproject_path`` is the file path; ``Path`` is the canonical
        cache key (no ``str(...)`` proxy round-trip).
        """
        if not pyproject_path.is_file():
            return {}
        payload_result = u.Cli.toml_read_json(pyproject_path)
        if payload_result.failure:
            return {}
        validated = FlextInfraUtilitiesPyproject.validate_infra_payload(
            payload_result.value
        )
        return validated if validated is not None else {}

    @staticmethod
    def normalized_toml_payload(document: t.Cli.TomlDocument) -> t.JsonMapping:
        """Return one TOML document normalized through the infra adapter."""
        payload = u.Cli.toml_as_mapping(document)
        if not payload:
            return {}
        validated = FlextInfraUtilitiesPyproject.validate_infra_payload(payload)
        return validated if validated is not None else {}

    @staticmethod
    def tool_flext_meta(project_root: Path) -> t.JsonMapping:
        """Return the normalized ``tool.flext`` table from a project root."""
        payload = FlextInfraUtilitiesPyproject.pyproject_payload(
            project_root / c.Infra.PYPROJECT_FILENAME
        )
        tool = payload.get(c.Infra.TOOL)
        if not isinstance(tool, dict):
            return {}
        flext = tool.get("flext")
        return flext if isinstance(flext, dict) else {}

    @staticmethod
    def docs_meta_from_payload(payload: t.JsonMapping) -> t.JsonMapping:
        """Extract ``tool.flext.docs`` metadata from an already-parsed payload."""
        tool = payload.get(c.Infra.TOOL)
        if not isinstance(tool, dict):
            return {}
        flext = tool.get("flext")
        if not isinstance(flext, dict):
            return {}
        docs = flext.get("docs")
        return docs if isinstance(docs, dict) else {}

    @staticmethod
    def project_name_from_payload(entry: Path, payload: t.JsonMapping) -> str:
        """Return the declared project name from ``[project].name``."""
        project_section = payload.get("project")
        if not isinstance(project_section, dict):
            msg = f"{entry}: missing [project] table in pyproject.toml"
            raise TypeError(msg)
        raw_name = project_section.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            msg = f"{entry}: missing or empty [project].name in pyproject.toml"
            raise ValueError(msg)
        return raw_name.strip()

    @staticmethod
    def package_name_from_payload(
        project_root: Path, payload: t.JsonMapping, docs_meta: t.JsonMapping
    ) -> str:
        """Return the primary package name using pre-loaded pyproject payload."""
        configured = docs_meta.get("package_name")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        current: t.JsonMapping | None = payload
        for key in (c.Infra.TOOL, "hatch", "build", "targets", "wheel"):
            if current is None:
                break
            candidate = current.get(key)
            current = candidate if isinstance(candidate, dict) else None
        packages = current.get("packages") if current is not None else None
        if isinstance(packages, list):
            for item in packages:
                package_path = Path(str(item).strip())
                if package_path.parts:
                    package_parts: tuple[str, ...] = package_path.parts
                    return package_parts[-1]
        src_dir = project_root / c.Infra.DEFAULT_SRC_DIR
        if src_dir.is_dir():
            for child in sorted(src_dir.iterdir()):
                if child.is_dir() and (child / c.Infra.INIT_PY).is_file():
                    child_path: Path = child
                    return child_path.name
        project_name = FlextInfraUtilitiesPyproject.project_name_from_payload(
            project_root, payload
        )
        if project_name.startswith(c.Infra.PKG_PREFIX_HYPHEN):
            msg = (
                f"{project_root}: cannot resolve package name — "
                "no [tool.flext.docs].package_name, no hatch wheel packages, "
                "and no src/<pkg>/__init__.py present"
            )
            raise ValueError(msg)
        return ""

    @staticmethod
    def project_package_name(project_root: Path) -> str:
        """Return the primary Python package name for a project root."""
        payload = FlextInfraUtilitiesPyproject.pyproject_payload(
            project_root / c.Infra.PYPROJECT_FILENAME
        )
        docs_meta = FlextInfraUtilitiesPyproject.docs_meta_from_payload(payload)
        return FlextInfraUtilitiesPyproject.package_name_from_payload(
            project_root, payload, docs_meta
        )

    @staticmethod
    @cache
    def workspace_project_paths(repository_root: Path) -> t.StrSequence:
        """Return project paths declared by this directory's own ``.gitmodules``.

        A missing file denotes a standalone project and therefore an empty
        sequence. A malformed declaration is an invalid workspace contract and
        remains a loud error; no pyproject table or parent directory is used as
        an alternate topology source.
        """
        declared = FlextInfraUtilitiesGit.git_declared_submodule_paths(repository_root)
        if declared.failure:
            msg = declared.error or f"invalid workspace topology: {repository_root}"
            raise ValueError(msg)
        return tuple(path.as_posix() for path in declared.value)


__all__: list[str] = ["FlextInfraUtilitiesPyproject"]
