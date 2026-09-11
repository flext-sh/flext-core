"""Project discovery helpers for enforcement dispatch."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

from flext_core import p as core_p
from flext_core import r
from flext_tests import c, p, t
from flext_tests.utilities import u


def _load_infra_report(
    workspace_root: Path, *, project_names: t.StrSequence
) -> p.Result[p.AttributeProbe]:
    """Return a workspace enforcement report when available."""
    if not project_names:
        return r[p.AttributeProbe].fail("no project names provided")
    import_result = u.try_(
        lambda: import_module("flext_infra.refactor.namespace_enforcer"),
        catch=ImportError,
        op_name="import flext_infra namespace enforcer",
    )
    if import_result.failure:
        return r[p.AttributeProbe].fail(
            import_result.error or "import flext_infra namespace enforcer failed"
        )
    refactor = import_result.value
    enforcer_cls = getattr(refactor, "FlextInfraNamespaceEnforcer", None)
    if enforcer_cls is None:
        return r[p.AttributeProbe].fail("FlextInfraNamespaceEnforcer not found")
    return u.try_(
        lambda: enforcer_cls(workspace_root=workspace_root).enforce(
            project_names=project_names
        ),
        catch=c.EXC_BROAD_RUNTIME,
        op_name="run flext_infra namespace enforcement",
    )


def _item_path(item: pytest.Item) -> Path | None:
    """Return the filesystem path represented by one collected pytest item."""
    path_value = getattr(item, "path", None)
    if isinstance(path_value, Path):
        return path_value.resolve()
    fspath = getattr(item, "fspath", None)
    if fspath is None:
        return None
    return Path(str(fspath)).resolve()


def _project_name_for_path(*, path: Path, workspace_root: Path) -> core_p.Result[str]:
    """Return the owning FLEXT project name for one workspace path."""
    relative_result = u.try_(
        lambda: path.relative_to(workspace_root),
        catch=ValueError,
        op_name="resolve relative workspace path",
    )
    if relative_result.failure:
        return r[str].fail(
            relative_result.error or "path is not relative to workspace root"
        )
    relative_path = relative_result.value
    if not relative_path.parts:
        return r[str].fail("path has no parts relative to workspace root")
    project_name = relative_path.parts[0]
    project_root = workspace_root / project_name
    if not (
        project_name.startswith("flext-")
        and project_root.is_dir()
        and (project_root / "pyproject.toml").is_file()
    ):
        return r[str].fail(f"{project_name} is not a recognized FLEXT project")
    return r[str].ok(project_name)


def _project_name_for_item(*, item: pytest.Item, workspace_root: Path) -> str | None:
    """Return the owning FLEXT project name for one collected item."""
    item_path = _item_path(item)
    if item_path is None:
        return None
    project_name_result = _project_name_for_path(
        path=item_path, workspace_root=workspace_root
    )
    return project_name_result.value if project_name_result.success else None


def _collected_project_names(
    *, items: t.SequenceOf[pytest.Item], workspace_root: Path
) -> t.StrSequence:
    """Return sorted FLEXT project names represented by collected pytest items."""
    project_names = {
        project_name
        for item in items
        if (
            project_name := _project_name_for_item(
                item=item, workspace_root=workspace_root
            )
        )
        is not None
    }
    return tuple(sorted(project_names))


def _validator_target_for_item(
    *, item: pytest.Item, workspace_root: Path
) -> Path | None:
    """Return the validation target represented by one collected item."""
    item_path = _item_path(item)
    if item_path is None:
        return None
    project_name_result = _project_name_for_path(
        path=item_path, workspace_root=workspace_root
    )
    if project_name_result.success:
        return workspace_root / project_name_result.value
    return item_path


def _collected_validator_targets(
    *, items: t.SequenceOf[pytest.Item], workspace_root: Path
) -> t.SequenceOf[Path]:
    """Return sorted validation targets represented by collected pytest items."""
    targets = {
        target
        for item in items
        if (
            target := _validator_target_for_item(
                item=item, workspace_root=workspace_root
            )
        )
        is not None
    }
    return tuple(sorted(targets))


collected_project_names = _collected_project_names
collected_validator_targets = _collected_validator_targets
load_infra_report = _load_infra_report
project_name_for_path = _project_name_for_path

__all__: list[str] = [
    "_collected_project_names",
    "_collected_validator_targets",
    "_load_infra_report",
    "_project_name_for_path",
    "collected_project_names",
    "collected_validator_targets",
    "load_infra_report",
    "project_name_for_path",
]
