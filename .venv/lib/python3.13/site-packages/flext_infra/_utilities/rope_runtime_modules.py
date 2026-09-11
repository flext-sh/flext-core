"""Rope project, module and import factory boundary methods."""

from __future__ import annotations

from collections.abc import Iterable

from flext_infra._utilities.rope_runtime_base import FlextInfraUtilitiesRopeRuntimeBase
from flext_infra.protocols import p
from flext_infra.typings import t


class FlextInfraUtilitiesRopeRuntimeModules(FlextInfraUtilitiesRopeRuntimeBase):
    """Load Rope project/module/import objects behind protocols."""

    @classmethod
    def new_project(
        cls,
        root: str,
        *,
        ropefolder: str,
        save_objectdb: bool,
        ignored_resources: t.SequenceOf[str],
        source_folders: t.SequenceOf[str],
    ) -> t.Infra.RopeProject:
        project_factory = cls._runtime_callable("rope.base.project", "Project")
        # flext-i6nq.10: FLEXT owns writes; disable Rope's leaking Git subprocess.
        fscommands_factory = cls._runtime_callable(
            "rope.base.fscommands", "FileSystemCommands"
        )
        project = project_factory(
            root,
            fscommands=fscommands_factory(),
            ropefolder=ropefolder,
            save_objectdb=save_objectdb,
            ignored_resources=list(ignored_resources),
            source_folders=list(source_folders),
        )
        if not isinstance(project, p.Infra.RopeProject):
            msg = "rope Project does not satisfy p.Infra.RopeProject"
            raise TypeError(msg)
        return project

    @classmethod
    def module_imports_for_pymodule(
        cls, rope_project: t.Infra.RopeProject, pymodule: t.Infra.RopePyModule
    ) -> t.Infra.RopeModuleImports:
        loader = cls._runtime_callable(
            "rope.refactor.importutils", "get_module_imports"
        )
        result = loader(rope_project, pymodule)
        if not isinstance(result, p.Infra.RopeModuleImports):
            msg = "rope get_module_imports returned invalid module imports"
            raise TypeError(msg)
        return result

    @classmethod
    def get_string_module(
        cls,
        rope_project: t.Infra.RopeProject,
        source: str,
        *,
        resource: t.Infra.RopeResource | None = None,
    ) -> t.Infra.RopePyModule:
        loader = cls._runtime_callable("rope.base.libutils", "get_string_module")
        pymodule = loader(rope_project, source, resource=resource, force_errors=True)
        if not isinstance(pymodule, p.Infra.RopePyModule):
            msg = "rope get_string_module returned non-PyModule"
            raise TypeError(msg)
        return pymodule

    @classmethod
    def import_organizer(
        cls, rope_project: t.Infra.RopeProject
    ) -> p.Infra.RopeImportOrganizer:
        organizer_factory = cls._runtime_callable(
            "rope.refactor.importutils", "ImportOrganizer"
        )
        organizer = organizer_factory(rope_project)
        if not isinstance(organizer, p.Infra.RopeImportOrganizer):
            msg = "rope ImportOrganizer does not satisfy p.Infra.RopeImportOrganizer"
            raise TypeError(msg)
        return organizer

    @classmethod
    def runtime_find_occurrences(
        cls,
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        offset: int,
        *,
        resources: t.SequenceOf[t.Infra.RopeResource] | None,
        in_hierarchy: bool,
    ) -> t.SequenceOf[t.Infra.RopeLocation]:
        finder = cls._runtime_callable("rope.contrib.findit", "find_occurrences")
        raw_locations = finder(
            rope_project,
            resource,
            offset,
            resources=resources,
            in_hierarchy=in_hierarchy,
        )
        if not isinstance(raw_locations, Iterable):
            msg = "rope find_occurrences returned non-iterable locations"
            raise TypeError(msg)
        return tuple(
            location
            for location in raw_locations
            if isinstance(location, p.Infra.RopeLocation)
        )

    @classmethod
    def from_import(
        cls,
        module_name: str,
        level: int,
        names_and_aliases: t.SequenceOf[tuple[str, str | None]],
    ) -> t.Infra.RopeFromImport:
        from_import_factory = cls._runtime_callable(
            "rope.refactor.importutils.importinfo", "FromImport"
        )
        from_import = from_import_factory(module_name, level, list(names_and_aliases))
        if not isinstance(from_import, p.Infra.RopeFromImport):
            msg = "rope FromImport does not satisfy p.Infra.RopeFromImport"
            raise TypeError(msg)
        return from_import


__all__: list[str] = ["FlextInfraUtilitiesRopeRuntimeModules"]
