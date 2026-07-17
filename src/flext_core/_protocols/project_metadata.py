"""Structural project metadata contracts exposed on ``p``."""

from __future__ import annotations

from typing import Protocol, runtime_checkable, TYPE_CHECKING

from flext_core._protocols.base import FlextProtocolsBase as pb


if TYPE_CHECKING:
    from pathlib import Path


# NOTE (multi-agent, mro-wkii.17.23 / agent: uv_overlay_owner): interfaces
# describe canonical model identities without transporting mappings or copies.
class FlextProtocolsProjectMetadata:
    """Protocols for project metadata consumed across FLEXT layers."""

    @runtime_checkable
    class ProjectAuthor(pb.BaseModel, Protocol):
        """PEP 621 author fields."""

        @property
        def name(self) -> str: ...

        @property
        def email(self) -> str: ...

    @runtime_checkable
    class ProjectUrls(pb.BaseModel, Protocol):
        """Canonical PEP 621 URL fields."""

        @property
        def homepage(self) -> str: ...

        @property
        def documentation(self) -> str: ...

        @property
        def repository(self) -> str: ...

    @runtime_checkable
    class Project(pb.BaseModel, Protocol):
        """PEP 621 project fields consumed by services."""

        @property
        def name(self) -> str: ...

        @property
        def version(self) -> str: ...

        @property
        def description(self) -> str: ...

        @property
        def requires_python(self) -> str: ...

        @property
        def dependencies(self) -> tuple[str, ...]: ...

        @property
        def authors(
            self,
        ) -> tuple[FlextProtocolsProjectMetadata.ProjectAuthor, ...]: ...

        @property
        def urls(self) -> FlextProtocolsProjectMetadata.ProjectUrls: ...

        @property
        def classifiers(self) -> tuple[str, ...]: ...

        @property
        def keywords(self) -> tuple[str, ...]: ...

    @runtime_checkable
    class ProjectToolFlextProject(pb.BaseModel, Protocol):
        """Project naming policy fields."""

        @property
        def class_stem_override(self) -> str | None: ...

    @runtime_checkable
    class ProjectToolFlextDocs(pb.BaseModel, Protocol):
        """Documentation policy fields."""

        @property
        def package_name(self) -> str | None: ...

        @property
        def project_class(self) -> str: ...

        @property
        def site_title(self) -> str | None: ...

        @property
        def exclude_docs(self) -> tuple[str, ...]: ...

    @runtime_checkable
    class ProjectToolFlextWorkspace(pb.BaseModel, Protocol):
        """Workspace attachment policy fields."""

        @property
        def attached(self) -> bool: ...

    @runtime_checkable
    class ProjectToolFlext(pb.BaseModel, Protocol):
        """Validated FLEXT project policy."""

        @property
        def project(self) -> FlextProtocolsProjectMetadata.ProjectToolFlextProject: ...

        @property
        def docs(self) -> FlextProtocolsProjectMetadata.ProjectToolFlextDocs: ...

        @property
        def workspace(
            self,
        ) -> FlextProtocolsProjectMetadata.ProjectToolFlextWorkspace: ...

    @runtime_checkable
    class ProjectMetadata(pb.BaseModel, Protocol):
        """Canonical retained project metadata aggregate."""

        @property
        def root(self) -> Path: ...

        @property
        def package_name(self) -> str: ...

        @property
        def class_stem(self) -> str: ...

        @property
        def project(self) -> FlextProtocolsProjectMetadata.Project: ...

        @property
        def flext(self) -> FlextProtocolsProjectMetadata.ProjectToolFlext: ...


__all__: list[str] = ["FlextProtocolsProjectMetadata"]
