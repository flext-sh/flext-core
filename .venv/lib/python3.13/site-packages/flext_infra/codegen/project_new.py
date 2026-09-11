"""``codegen new`` — create a project through the canonical conform pipeline.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

# New file per operator live
# order (ULW). ctx via u.derive_class_stem (no parallel detection, ADR-005 §9);
# accessor typing/config+settings symmetry fixed in templates in the same lane.
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, override

from flext_core import r
from flext_infra import c, config, m, u
from flext_infra.base import s
from flext_infra.codegen.conform import FlextInfraCodegenConform

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraCodegenProjectNew(s[m.Infra.CodegenResult]):
    """Create a new FLEXT project (internal member or external standalone)."""

    name: Annotated[
        str,
        m.Field(
            min_length=1,
            description="Distribution name in kebab-case (e.g. flext-demo / acme-demo).",
        ),
    ]
    kind: Annotated[
        c.Infra.ProjectKind,
        m.Field(description="Project kind: internal (monorepo member) or external."),
    ]
    output_root: Annotated[
        Path, m.Field(description="Directory that becomes the generated project root.")
    ]
    package_name: Annotated[
        str, m.Field(description="Python package name (default: name with '-'→'_').")
    ] = ""
    # Field renamed
    # ``namespace``→``project_namespace`` to avoid colliding with the inherited
    # base field ``target_namespace`` (alias ``namespace``); CLI flag stays ``--ns``.
    project_namespace: Annotated[
        str,
        m.Field(
            alias="ns",
            description="Facade namespace slot (default: class stem minus 'Flext').",
        ),
    ] = ""
    description: Annotated[
        str, m.Field(default="", description="Project description (default: derived).")
    ] = ""
    provider: Annotated[
        str, m.Field(min_length=1, description="Configured Git provider key.")
    ]
    repository_url: Annotated[
        str, m.Field(description="Canonical Git clone URL for the new repository.")
    ] = ""
    beads_workspace: Annotated[
        str, m.Field(min_length=1, description="Explicit Beads workspace identity.")
    ]
    beads_database: Annotated[
        str, m.Field(min_length=1, description="Explicit Beads database identity.")
    ]
    beads_issue_prefix: Annotated[
        str, m.Field(min_length=1, description="Explicit Beads issue prefix.")
    ]
    license: Annotated[
        str, m.Field(min_length=1, description="SPDX project license identifier.")
    ]
    author_name: Annotated[
        str, m.Field(min_length=1, description="Author/maintainer display name.")
    ]
    author_email: Annotated[
        str, m.Field(min_length=3, description="Author/maintainer email.")
    ]
    upstream: Annotated[str, m.Field(description="Upstream facade module (flext_cli).")]
    year: Annotated[int, m.Field(ge=2025, description="Deterministic copyright year.")]

    @override
    def execute(self) -> p.Result[m.Infra.CodegenResult]:
        """Build one typed manifest and delegate all output to conform."""
        if self.effective_dry_run:
            return r[m.Infra.CodegenResult].fail("codegen new requires apply mode")
        provider = next(
            (
                item
                for item in config.Infra.codegen.providers
                if item.name == self.provider
            ),
            None,
        )
        # The URL comes from the caller or is derived from the provider
        # contract. flext-infra keeps no catalog of existing projects to
        # consult, so scaffolding a new project needs no prior knowledge of it.
        package_name = self.package_name or self.name.replace("-", "_")
        class_stem = u.derive_class_stem(self.name)
        derived_namespace = class_stem.removeprefix("Flext")
        project_namespace = self.project_namespace or derived_namespace or class_stem
        alias = u.Infra.package_alias(package_name=package_name)
        provider_url = (
            f"{provider.base_url}/{self.name}.git" if provider is not None else ""
        )
        repository_url = self.repository_url or provider_url
        if not repository_url:
            return r[m.Infra.CodegenResult].fail(
                f"repository URL is required for provider: {self.provider}"
            )
        if provider is None:
            return r[m.Infra.CodegenResult].fail(
                f"repository branch is required for provider: {self.provider}"
            )
        repository_page = repository_url.removesuffix(".git")
        repository = m.Infra.RepositoryRef(
            name=self.name,
            distribution=self.name,
            provider=self.provider,
            url=repository_url,
            path=Path(),
            role=c.Infra.MakeProfile.STANDALONE,
            state=c.Infra.RepositoryState.ACTIVE,
            checkout=c.Infra.CheckoutKind.INDEPENDENT,
            codegen=c.Infra.CodegenKind.CONFORM,
            package=True,
            editable=False,
            read_only=False,
        )
        workspace = m.Infra.WorkspaceSpec(
            name=self.beads_workspace,
            beads=m.Infra.BeadsProjectSpec(
                version=c.Infra.BEADS_CONFIG_VERSION,
                workspace=self.beads_workspace,
                database=self.beads_database,
                issue_prefix=self.beads_issue_prefix,
            ),
            repository=repository,
            project=m.Infra.ProjectSpec(
                package_name=package_name,
                class_stem=class_stem,
                namespace=project_namespace,
                constant_name=self.name,
                namespace_attribute=alias,
                alias=alias,
                environment_prefix=f"{package_name.upper()}_",
                description=(
                    self.description
                    or f"{class_stem} — FLEXT typed integration package"
                ),
                license=self.license,
                author_name=self.author_name,
                author_email=self.author_email,
                upstream=self.upstream,
                homepage=repository_page,
                documentation=repository_page,
                repository_root_rel=".",
                year=self.year,
            ),
        )
        request = m.Infra.CodegenConformRequest(
            root=self.output_root.expanduser().resolve(),
            scope=c.Infra.CodegenConformScope.SELF,
            mode=c.Infra.CodegenConformMode.APPLY,
        )
        return FlextInfraCodegenConform.execute_request(
            request, initial_workspace=workspace
        )


__all__: list[str] = ["FlextInfraCodegenProjectNew"]
