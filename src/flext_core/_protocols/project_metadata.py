"""Structural project metadata contracts exposed on ``p``.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols.base import FlextProtocolsBase

if TYPE_CHECKING:
    from collections.abc import Iterable
    from importlib.metadata import Distribution, DistributionFinder
    from pathlib import Path, PurePosixPath

    from flext_core import t


# NOTE (multi-agent, mro-wkii.17.23 / agent: uv_overlay_owner): interfaces
# describe canonical model identities without transporting mappings or copies.
class FlextProtocolsProjectMetadata:
    """Protocols for project metadata consumed across FLEXT layers."""

    @runtime_checkable
    class ProjectAuthor(FlextProtocolsBase.Model, Protocol):
        """PEP 621 author fields."""

        @property
        def name(self) -> str: ...

        @property
        def email(self) -> str: ...

    @runtime_checkable
    class ProjectUrls(FlextProtocolsBase.Model, Protocol):
        """Canonical PEP 621 URL fields."""

        @property
        def homepage(self) -> str: ...

        @property
        def documentation(self) -> str: ...

        @property
        def repository(self) -> str: ...

    @runtime_checkable
    class Project(FlextProtocolsBase.Model, Protocol):
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
        def dependencies(self) -> t.VariadicTuple[str]: ...

        @property
        def authors(
            self,
        ) -> tuple[FlextProtocolsProjectMetadata.ProjectAuthor, ...]: ...

        @property
        def urls(self) -> FlextProtocolsProjectMetadata.ProjectUrls: ...

        @property
        def classifiers(self) -> t.VariadicTuple[str]: ...

        @property
        def keywords(self) -> t.VariadicTuple[str]: ...

    @runtime_checkable
    class ProjectToolFlextProject(FlextProtocolsBase.Model, Protocol):
        """Project naming policy fields."""

        @property
        def class_stem_override(self) -> str | None: ...

    @runtime_checkable
    class ProjectToolFlextReadmeSection(FlextProtocolsBase.Model, Protocol):
        """One ordered project README section declaration."""

        @property
        def id(self) -> str: ...

        @property
        def title(self) -> str: ...

        @property
        def content(self) -> str | None: ...

        @property
        def include(self) -> PurePosixPath | None: ...

    @runtime_checkable
    class ProjectToolFlextDocs(FlextProtocolsBase.Model, Protocol):
        """Documentation policy fields."""

        @property
        def package_name(self) -> str | None: ...

        @property
        def project_class(self) -> str: ...

        @property
        def site_title(self) -> str | None: ...

        @property
        def exclude_docs(self) -> t.VariadicTuple[str]: ...

        @property
        def readme_sections(
            self,
        ) -> tuple[
            FlextProtocolsProjectMetadata.ProjectToolFlextReadmeSection,
            ...,
        ]: ...

    @runtime_checkable
    class ProjectToolFlextWorkspace(FlextProtocolsBase.Model, Protocol):
        """Workspace attachment policy fields."""

        @property
        def attached(self) -> bool: ...

    @runtime_checkable
    class ProjectToolFlext(FlextProtocolsBase.Model, Protocol):
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
    class ProjectMetadata(FlextProtocolsBase.Model, Protocol):
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

    @runtime_checkable
    class DistributionSource(Protocol):
        """A ``sys.meta_path`` entry that reports installed distributions.

        Matches finder instances and finder classes alike (``PathFinder`` sits
        on ``sys.meta_path`` as a class), exactly as ``importlib.metadata``
        discovers them.
        """

        def find_distributions(
            self,
            context: DistributionFinder.Context = ...,
            /,
        ) -> Iterable[Distribution]: ...


__all__: list[str] = ["FlextProtocolsProjectMetadata"]
