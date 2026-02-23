from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import structlog
from flext_core import r
from tomlkit.toml_document import TOMLDocument

from flext_infra.constants import ic
from flext_infra.discovery import DiscoveryService
from flext_infra.toml_io import TomlService

logger = structlog.get_logger(__name__)

FLEXT_DEPS_DIR = ".flext-deps"

_PEP621_PATH_DEP_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)\s*@\s*(?:file:)?(?P<path>.+)$"
)
_PEP621_NAME_RE = re.compile(r"^\s*(?P<name>[A-Za-z0-9_.-]+)")


def _workspace_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / ".gitmodules").exists():
            return candidate
    return here.parents[4]


ROOT = _workspace_root()


def detect_mode(project_root: Path) -> str:
    for candidate in (project_root, *project_root.parents):
        if (candidate / ".gitmodules").exists():
            return "workspace"
    return "standalone"


def extract_dep_name(raw_path: str) -> str:
    normalized = raw_path.strip().removeprefix("./")
    for prefix in (f"{FLEXT_DEPS_DIR}/", "../"):
        normalized = normalized.removeprefix(prefix)
    return normalized


def _target_path(dep_name: str, *, is_root: bool, mode: str) -> str:
    if mode == "workspace":
        return dep_name if is_root else f"../{dep_name}"
    return f"{FLEXT_DEPS_DIR}/{dep_name}"


def _extract_requirement_name(entry: str) -> str | None:
    if " @ " in entry:
        match = _PEP621_PATH_DEP_RE.match(entry)
        if match:
            return match.group("name")
    match = _PEP621_NAME_RE.match(entry)
    if not match:
        return None
    return match.group("name")


def _rewrite_pep621(
    doc: TOMLDocument,
    *,
    is_root: bool,
    mode: str,
    internal_names: set[str],
) -> list[str]:
    project = doc.get("project")
    if not project or not isinstance(project, dict):
        return []
    deps = project.get("dependencies")
    if not isinstance(deps, list):
        return []

    changes: list[str] = []
    for index, item in enumerate(deps):
        if not isinstance(item, str):
            continue

        marker = ""
        requirement_part = item
        if ";" in item:
            requirement_part, marker_part = item.split(";", 1)
            marker = f" ;{marker_part}"

        dep_name = _extract_requirement_name(requirement_part)
        if not dep_name or dep_name not in internal_names:
            continue

        if " @ " in requirement_part:
            match = _PEP621_PATH_DEP_RE.match(requirement_part)
            if not match:
                continue
            raw_path = match.group("path").strip()
            dep_name = extract_dep_name(raw_path)

        new_path = _target_path(dep_name, is_root=is_root, mode=mode)
        path_prefix = "./" if is_root else ""
        new_entry = f"{dep_name} @ {path_prefix}{new_path}{marker}"
        if item != new_entry:
            changes.append(f"  PEP621: {item} -> {new_entry}")
            deps[index] = new_entry
    return changes


def _rewrite_poetry(doc: TOMLDocument, *, is_root: bool, mode: str) -> list[str]:
    tool = doc.get("tool")
    if not isinstance(tool, dict):
        return []
    poetry = tool.get("poetry")
    if not isinstance(poetry, dict):
        return []
    deps = poetry.get("dependencies")
    if not isinstance(deps, dict):
        return []

    changes: list[str] = []
    for dep_key, value in deps.items():
        if not isinstance(value, dict) or "path" not in value:
            continue
        raw_path = value["path"]
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        dep_name = extract_dep_name(raw_path)
        new_path = _target_path(dep_name, is_root=is_root, mode=mode)
        if raw_path != new_path:
            changes.append(f"  Poetry: {dep_key}.path = {raw_path!r} -> {new_path!r}")
            value["path"] = new_path
    return changes


def rewrite_dep_paths(
    pyproject_path: Path,
    *,
    mode: str,
    internal_names: set[str],
    is_root: bool = False,
    dry_run: bool = False,
) -> r[list[str]]:
    toml_service = TomlService()
    doc_result = toml_service.read_document(pyproject_path)
    if doc_result.is_failure:
        return r[list[str]].fail(doc_result.error or "failed to read TOML document")

    doc = doc_result.value
    changes = _rewrite_pep621(
        doc,
        is_root=is_root,
        mode=mode,
        internal_names=internal_names,
    )
    changes += _rewrite_poetry(doc, is_root=is_root, mode=mode)

    if changes and not dry_run:
        write_result = toml_service.write_document(pyproject_path, doc)
        if write_result.is_failure:
            return r[list[str]].fail(write_result.error or "failed to write TOML")

    return r[list[str]].ok(changes)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite internal FLEXT dependency paths for workspace/standalone mode."
    )
    _ = parser.add_argument(
        "--mode",
        choices=["workspace", "standalone", "auto"],
        default="auto",
        help="Target mode (default: auto-detect)",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without writing",
    )
    _ = parser.add_argument(
        "--project",
        action="append",
        dest="projects",
        metavar="DIR",
        help="Specific project(s) to process (default: all)",
    )
    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = detect_mode(ROOT)
        print(f"[sync-dep-paths] auto-detected mode: {mode}")

    total_changes = 0
    toml_service = TomlService()

    internal_names: set[str] = set()
    root_pyproject = ROOT / ic.Files.PYPROJECT_FILENAME
    if root_pyproject.exists():
        root_data_result = toml_service.read(root_pyproject)
        if root_data_result.is_success:
            root_project = root_data_result.value.get("project")
            if isinstance(root_project, dict):
                root_name = root_project.get("name")
                if isinstance(root_name, str) and root_name:
                    internal_names.add(root_name)

    if not args.projects:
        if root_pyproject.exists():
            changes_result = rewrite_dep_paths(
                root_pyproject,
                mode=mode,
                internal_names=internal_names,
                is_root=True,
                dry_run=args.dry_run,
            )
            if changes_result.is_failure:
                logger.error(
                    "sync_dep_paths_root_failed",
                    pyproject=str(root_pyproject),
                    error=changes_result.error,
                )
                return 1
            changes = changes_result.value
            if changes:
                prefix = "[DRY-RUN] " if args.dry_run else ""
                print(f"{prefix}{root_pyproject}:")
                for change in changes:
                    print(change)
                total_changes += len(changes)

    discover_result = DiscoveryService().discover_projects(ROOT)
    if discover_result.is_failure:
        logger.error(
            "sync_dep_paths_discovery_failed",
            root=str(ROOT),
            error=discover_result.error,
        )
        return 1

    all_project_dirs = [project.path for project in discover_result.value]
    if args.projects:
        project_dirs = [ROOT / project for project in args.projects]
    else:
        project_dirs = all_project_dirs

    for project_dir in all_project_dirs:
        pyproject = project_dir / ic.Files.PYPROJECT_FILENAME
        if not pyproject.exists():
            continue
        data_result = toml_service.read(pyproject)
        if data_result.is_failure:
            continue
        project_obj = data_result.value.get("project")
        if not isinstance(project_obj, dict):
            continue
        project_name = project_obj.get("name")
        if isinstance(project_name, str) and project_name:
            internal_names.add(project_name)

    for project_dir in project_dirs:
        pyproject = project_dir / ic.Files.PYPROJECT_FILENAME
        if not pyproject.exists():
            continue
        data_result = toml_service.read(pyproject)
        if data_result.is_failure:
            continue
        project_obj = data_result.value.get("project")
        if not isinstance(project_obj, dict):
            continue
        project_name = project_obj.get("name")
        if isinstance(project_name, str) and project_name:
            internal_names.add(project_name)

    for project_dir in sorted(project_dirs):
        pyproject = project_dir / ic.Files.PYPROJECT_FILENAME
        if not pyproject.exists():
            continue
        changes_result = rewrite_dep_paths(
            pyproject,
            mode=mode,
            internal_names=internal_names,
            is_root=False,
            dry_run=args.dry_run,
        )
        if changes_result.is_failure:
            logger.error(
                "sync_dep_paths_project_failed",
                pyproject=str(pyproject),
                error=changes_result.error,
            )
            return 1
        changes = changes_result.value
        if changes:
            prefix = "[DRY-RUN] " if args.dry_run else ""
            print(f"{prefix}{pyproject}:")
            for change in changes:
                print(change)
            total_changes += len(changes)

    if total_changes == 0:
        print(
            "[sync-dep-paths] No changes needed - all paths already match target mode."
        )
    else:
        action = "would change" if args.dry_run else "changed"
        print(f"\n[sync-dep-paths] {action} {total_changes} path(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "FLEXT_DEPS_DIR",
    "detect_mode",
    "extract_dep_name",
    "main",
    "rewrite_dep_paths",
]
