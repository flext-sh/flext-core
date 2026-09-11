"""Pure Pydantic config and codegen contracts for flext-infra.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self

from flext_cli import m, u
from flext_infra import t
from flext_infra._constants.codegen_project import FlextInfraConstantsCodegenProject
from flext_infra._constants.make import FlextInfraConstantsMake
from flext_infra._constants.release import FlextInfraConstantsRelease
from flext_infra._constants.validate import FlextInfraConstantsSharedInfra
from flext_infra._models._defaults import immutable_empty_mapping
from flext_infra._models.deps_tool_config import FlextInfraModelsDepsToolSettings
from flext_infra._models.layout import FlextInfraModelsLayout


class _ConfigContract(m.ContractModel):
    """Private declarative base for schema-loaded codegen records."""

    # Rendered file payloads are
    # byte contracts; Pydantic must never trim their final newline.
    model_config = m.ConfigDict(
        strict=False, frozen=True, extra="forbid", str_strip_whitespace=False
    )


def _tool_version_field(description: str) -> object:
    """Shared ``Annotated[t.NonEmptyStr, ...]`` metadata for one tool version.

    Every native-toolchain version field in ``ToolchainSpec`` and its
    ``ProjectRenderContext`` render mirror previously repeated an identical
    ``m.Field(description=...)`` shape, differing only in the description
    text -- a structural clone SonarCloud's duplication detector flags as one
    family regardless of the literal string. One owned factory collapses
    every call site to this single declaration (SSOT, DRY).
    """
    return m.Field(description=description)


class FlextInfraConfigModels:
    """Field-only models for config loading and codegen plans."""

    # These models replace the former model-less workspace/make dictionaries.
    # YAML is accepted only at the flext-cli loading boundary and is immediately
    # model-validated here.

    class MiseToolSpec(_ConfigContract):
        """One exact mise backend selector and immutable version."""

        selector: Annotated[
            t.NonEmptyStr, m.Field(description="Canonical mise backend selector")
        ]
        version: Annotated[
            t.NonEmptyStr, m.Field(description="Exact tool version installed by mise")
        ]
        prerelease: Annotated[
            bool,
            m.Field(
                description="Whether mise may resolve prerelease versions for this tool"
            ),
        ] = False
        minimum_release_age: Annotated[
            t.NonEmptyStr | None,
            m.Field(
                description=(
                    "Per-tool release quarantine override passed to mise; absent "
                    "preserves mise's global security default"
                )
            ),
        ] = None

    class ProtectedMiseToolSpec(MiseToolSpec):
        """One fleet-owned mise distribution identity."""

        selector_patterns: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                min_length=1,
                description="Glob patterns identifying equivalent mise distributions",
            ),
        ]

        @u.model_validator(mode="after")
        def _validate_distribution_patterns(self) -> Self:
            """Require one unambiguous pattern set covering the canonical selector."""
            if len(set(self.selector_patterns)) != len(self.selector_patterns):
                msg = "protected mise selector_patterns must be unique"
                raise ValueError(msg)
            if not any(
                fnmatchcase(self.selector, pattern)
                for pattern in self.selector_patterns
            ):
                msg = (
                    "canonical mise selector is not covered by selector_patterns: "
                    f"{self.selector}"
                )
                raise ValueError(msg)
            return self

    class BeadsEndpointSpec(_ConfigContract):
        """Static network endpoint projected into Beads configuration."""

        host: Annotated[t.NonEmptyStr, m.Field(description="Beads server host")]
        port: Annotated[
            int,
            m.Field(
                ge=1,
                le=65535,
                description="Beads server TCP port declared by deployment",
            ),
        ]

    class BeadsToolSpec(ProtectedMiseToolSpec):
        """Canonical Beads distribution and Gas City projection contract."""

        endpoint_origin: Annotated[
            Literal["inherited_city"],
            m.Field(description="Gas City-owned endpoint inheritance mode"),
        ]
        endpoint_status: Annotated[
            Literal["verified"],
            m.Field(description="Canonical status for a managed-city inherited rig"),
        ]
        required_custom_types: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                min_length=1,
                description="Immutable custom bead types required by Gas City",
            ),
        ]

        @u.model_validator(mode="after")
        def _validate_required_custom_types(self) -> Self:
            """Reject ambiguous duplicate type declarations at the owner."""
            if len(set(self.required_custom_types)) != len(self.required_custom_types):
                msg = "beads required_custom_types must be unique"
                raise ValueError(msg)
            return self

    class MiseLockPlatformSpec(_ConfigContract):
        """Immutable download metadata for one tool platform."""

        checksum: Annotated[
            t.NonEmptyStr,
            m.Field(
                pattern=r"^sha256:[0-9a-f]{64}$",
                description="SHA-256 digest emitted by Mise",
            ),
        ]
        url: Annotated[
            t.NonEmptyStr,
            m.Field(
                pattern=r"^https://",
                description="Immutable HTTPS artifact URL emitted by Mise",
            ),
        ]
        url_api: Annotated[
            t.NonEmptyStr | None,
            m.Field(description="Optional immutable provider API URL"),
        ] = None
        provenance: Annotated[
            t.NonEmptyStr | None,
            m.Field(description="Optional artifact provenance mechanism"),
        ] = None
        provenance_verified: Annotated[
            bool | None,
            m.Field(description="Optional local provenance verification result"),
        ] = None

    class MiseLockToolSpec(_ConfigContract):
        """One resolved tool version and its platform artifacts."""

        version: Annotated[
            t.NonEmptyStr, m.Field(description="Resolved immutable tool version")
        ]
        backend: Annotated[
            t.NonEmptyStr, m.Field(description="Resolved Mise backend identity")
        ]
        specifiers: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                min_length=1, description="Source selectors resolved by this entry"
            ),
        ]
        platforms: Annotated[
            Mapping[t.NonEmptyStr, FlextInfraConfigModels.MiseLockPlatformSpec],
            m.Field(
                default_factory=dict, description="Resolved platform download metadata"
            ),
        ]

    class MiseLockSpec(_ConfigContract):
        """Typed shape of the generated Mise lockfile."""

        lockfile_version: Annotated[
            Literal[1], m.Field(description="Supported Mise lock schema version")
        ]
        tools: Annotated[
            Mapping[t.NonEmptyStr, tuple[FlextInfraConfigModels.MiseLockToolSpec, ...]],
            m.Field(description="Exactly resolved generated tool set"),
        ]

    class ToolchainSpec(_ConfigContract):
        """Language-runtime and native-tool versions shared by generated projects.

        Only the Python minor line ``python_version`` (e.g. ``3.13``) is
        declared for the language runtime. The environment resolves its newest
        compatible patch. The PEP 440 family requirement is derived, so a
        version-line bump touches exactly one value. uv is supplied by the caller
        environment. Python linters/type-checkers are NOT here: their floors live
        in pyproject and uv.lock owns the resolved versions. Native executables
        required by canonical Make gates are declared here for reproducible
        provisioning through mise.
        """

        python_version: Annotated[
            t.NonEmptyStr,
            m.Field(
                pattern=r"^[0-9]+\.[0-9]+$",
                description="Python major.minor line, e.g. '3.13'",
            ),
        ]
        uv_link_mode: Annotated[
            t.NonEmptyStr, m.Field(description="Portable uv installation link mode")
        ]
        uv_environments: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                description=(
                    "Marker expressions limiting the environments uv resolves "
                    "for the generated lock. Empty resolves every environment."
                )
            ),
        ] = ()
        dependency_constraints: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                description=(
                    "Fleet-wide PEP 508 resolution constraints projected to "
                    "[tool.uv] constraint-dependencies of every generated "
                    "pyproject. A member's own lock then resolves inside the "
                    "range every sibling can install, so the floor rewrite "
                    "(which raises floors to the locked version) can never "
                    "raise a floor above a cap another member declares. Empty "
                    "removes the key."
                )
            ),
        ] = ()
        dependency_cooldown_days: Annotated[
            int,
            m.Field(
                ge=1,
                le=90,
                description=(
                    "Supply-chain cooldown shared by uv resolution and "
                    "Dependabot version updates"
                ),
            ),
        ]
        dependency_cooldown_exclusions: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                description=(
                    "Packages explicitly exempted from cooldown for urgent "
                    "security floors"
                )
            ),
        ] = ()
        dependency_cooldown_overrides: Annotated[
            t.StrMapping,
            m.Field(
                default_factory=immutable_empty_mapping,
                description=(
                    "Per-package cooldown cutoffs, mapping package name to an "
                    "RFC 3339 timestamp. The shared window is a single date for "
                    "the whole resolution, so a project that legitimately "
                    "requires a floor published after it becomes unsatisfiable: "
                    "uv reports the requirement cannot be met and names "
                    "exclude-newer-package as the remedy. A boolean exemption "
                    "cannot express this, because the cutoff has to move to a "
                    "specific instant rather than be switched off, so the "
                    "override carries the timestamp."
                ),
            ),
        ]
        kubectl_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact kubectl version, e.g. '1.32.0'")
        ]
        helm_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact Helm version, e.g. '3.19.4'")
        ]
        kind_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact kind version, e.g. '0.31.0'")
        ]
        direnv_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Compatible direnv major.minor line")
        ]
        environment_path_prepends: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                default=(),
                description=(
                    "Extra directories the generated shell activation prepends "
                    "to PATH when they exist. Installation data expressed as "
                    "shell-expandable paths; empty by default so the engine "
                    "never names a specific tool installation."
                ),
            ),
        ] = ()
        uv_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Compatible uv major.minor line")
        ]
        qlty_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact qlty code-quality version")
        ]
        node_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Compatible Node.js major.minor line")
        ]
        jscpd_version: Annotated[
            t.NonEmptyStr,
            _tool_version_field("Exact jscpd duplication detector version"),
        ]
        taplo_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact Taplo formatter version")
        ]
        ast_grep_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact ast-grep analyzer version")
        ]
        gitleaks_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact Gitleaks scanner version")
        ]
        scc_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact scc code-counter version")
        ]
        kubeconform_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Compatible kubeconform minor line")
        ]
        go_version: Annotated[
            t.NonEmptyStr,
            _tool_version_field(
                "Exact Go runtime version; mise resolves go: backend "
                "selectors through it, so beads only installs when Go "
                "is a declared tool"
            ),
        ]
        mise_lock_platforms: Annotated[
            tuple[
                Literal[
                    "linux-x64",
                    "linux-arm64",
                    "linux-x64-musl",
                    "linux-arm64-musl",
                    "macos-x64",
                    "macos-arm64",
                    "windows-x64",
                ],
                ...,
            ],
            m.Field(
                min_length=1,
                description="Platforms materialized into the project mise lockfile",
            ),
        ]
        mise_lock_platform_exclusions: Annotated[
            Mapping[
                t.NonEmptyStr,
                tuple[
                    Literal[
                        "linux-x64",
                        "linux-arm64",
                        "linux-x64-musl",
                        "linux-arm64-musl",
                        "macos-x64",
                        "macos-arm64",
                        "windows-x64",
                    ],
                    ...,
                ],
            ],
            m.Field(
                default_factory=immutable_empty_mapping,
                description=(
                    "Explicit platforms a backend cannot represent in mise.lock"
                ),
            ),
        ]
        beads: Annotated[
            FlextInfraConfigModels.BeadsToolSpec,
            m.Field(description="Official Beads CLI installed through mise"),
        ]
        gascity: Annotated[
            FlextInfraConfigModels.ProtectedMiseToolSpec,
            m.Field(description="Gas City CLI (gc) installed through mise"),
        ]
        protected_mise_tools: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                min_length=1,
                description="Toolchain field names protected from alternate distributions",
            ),
        ]

        @u.model_validator(mode="after")
        def _validate_protected_mise_tools(self) -> Self:
            """Resolve every protected owner to the generic identity contract."""
            if len(set(self.protected_mise_tools)) != len(self.protected_mise_tools):
                msg = "protected_mise_tools must be unique"
                raise ValueError(msg)
            for owner in self.protected_mise_tools:
                if not isinstance(
                    getattr(self, owner, None),
                    FlextInfraConfigModels.ProtectedMiseToolSpec,
                ):
                    msg = f"protected_mise_tools references invalid owner: {owner}"
                    raise TypeError(msg)
            return self

        @u.model_validator(mode="after")
        def _validate_mise_lock_platforms(self) -> Self:
            """Reject duplicate lock targets before platform metadata generation."""
            if len(set(self.mise_lock_platforms)) != len(self.mise_lock_platforms):
                msg = "mise_lock_platforms must be unique"
                raise ValueError(msg)
            declared = set(self.mise_lock_platforms)
            for selector, exclusions in self.mise_lock_platform_exclusions.items():
                if len(set(exclusions)) != len(exclusions):
                    msg = f"mise lock platform exclusions must be unique: {selector}"
                    raise ValueError(msg)
                unknown = set(exclusions) - declared
                if unknown:
                    msg = (
                        "mise lock platform exclusions must be declared targets: "
                        f"{selector}"
                    )
                    raise ValueError(msg)
            return self

        @m.computed_field
        @property
        def python_required_version(self) -> str:
            """PEP 440 requirement spanning the configured Python minor line."""
            major, _, minor = self.python_version.partition(".")
            next_minor = int(minor) + 1
            return f">={self.python_version},<{major}.{next_minor}"

        @m.computed_field
        @property
        def python_selector(self) -> str:
            """Mise/pyenv-style selector for the configured Python minor line."""
            return self.python_version

        @m.computed_field
        @property
        def uv_exclude_newer(self) -> str:
            """Render the shared dependency cooldown in uv duration syntax."""
            return f"{self.dependency_cooldown_days} days"

    class ProviderSpec(_ConfigContract):
        """One GitHub organization and its mandatory branch policy."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Provider key")]
        organization: Annotated[
            t.NonEmptyStr, m.Field(description="GitHub organization")
        ]
        base_url: Annotated[t.NonEmptyStr, m.Field(description="GitHub HTTPS base URL")]
        branch: Annotated[t.NonEmptyStr, m.Field(description="Provider branch")]

    class RepositorySourceSpec(_ConfigContract):
        """Portable repository identity derived through one declared provider."""

        distribution: Annotated[
            t.NonEmptyStr, m.Field(description="Repository distribution name")
        ]
        provider: Annotated[
            t.NonEmptyStr, m.Field(description="Provider key owning URL and branch")
        ]

        @m.computed_field
        @property
        def internal_distribution_prefix(self) -> str:
            """Derive the internal distribution namespace from the owner name."""
            namespace, _, _ = self.distribution.partition("-")
            return f"{namespace}-"

    class BranchPolicySpec(_ConfigContract):
        """Global ancestry policy shared by every governed provider."""

        REQUIRED_TECHNICAL_PATTERNS: ClassVar[tuple[str, ...]] = (
            "__dolt_remote_info__",
            "dolt/*",
            "gh-readonly-queue/*",
        )
        technical_branch_patterns: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                description=(
                    "GitHub/Dolt technical branches excluded from ancestry validation"
                )
            ),
        ]
        governed_branch_patterns: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                min_length=1,
                description=(
                    "Development lines whose descent from the baseline is enforced. "
                    "Refs outside this allowlist are inventoried but never gated: "
                    "parked releases, snapshots and lane branches must not block."
                ),
            ),
        ]
        # Why (flext-1wjg1.16.34): both fields are live — the CI/docs templates
        # iterate ci_trigger_branches and the release/baseline resolution reads
        # integration_branch_preference — so the merge keeps both.
        ci_trigger_branches: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                min_length=1,
                description="Fleet branches receiving blocking generated CI triggers",
            ),
        ]
        integration_branch_preference: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                default=FlextInfraConstantsSharedInfra.INTEGRATION_BRANCH_PREFERENCE,
                min_length=1,
                description=(
                    "Ordered names tried against live Git to derive one "
                    "repository's integration baseline. A workspace that "
                    "integrates on a versioned line declares it here, so a "
                    "release name never has to be hardcoded in the package. "
                    "The provider default stays the last-resort fallback, "
                    "never the answer."
                ),
            ),
        ] = FlextInfraConstantsSharedInfra.INTEGRATION_BRANCH_PREFERENCE

        @u.model_validator(mode="after")
        def _validate_technical_patterns(self) -> Self:
            """Keep the global exclusion set exact and non-extensible."""
            if self.technical_branch_patterns != self.REQUIRED_TECHNICAL_PATTERNS:
                msg = (
                    "technical branch patterns must equal the canonical GitHub/Dolt "
                    f"set: {', '.join(self.REQUIRED_TECHNICAL_PATTERNS)}"
                )
                raise ValueError(msg)
            return self

    class GithubActionPinSpec(_ConfigContract):
        """One immutable GitHub Action reference from the codegen catalog."""

        repository: Annotated[
            t.NonEmptyStr, m.Field(description="GitHub owner/repository action name")
        ]
        version: Annotated[
            t.NonEmptyStr, m.Field(description="Human-readable upstream release tag")
        ]
        sha: Annotated[
            t.NonEmptyStr,
            m.Field(
                pattern=r"^[0-9a-f]{40}$",
                description="Immutable upstream action commit",
            ),
        ]

    class CiPrivateSubmoduleDeployKeySpec(_ConfigContract):
        """One read-only deploy key that unlocks a private workspace declared_repository in CI."""

        secret: Annotated[
            t.NonEmptyStr,
            m.Field(
                description="GitHub Actions secret name holding the deploy key PEM"
            ),
        ]
        submodule: Annotated[
            t.NonEmptyStr,
            m.Field(
                description="gitmodules submodule name (git config submodule.<name>.url)"
            ),
        ]
        path: Annotated[
            t.NonEmptyStr, m.Field(description="Checkout-relative submodule path")
        ]
        remote: Annotated[
            t.NonEmptyStr,
            m.Field(
                pattern=r"^git@github\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$",
                description="Canonical GitHub SSH clone URL without a Host alias",
            ),
        ]

    class CiPrivateSubmodulesSpec(_ConfigContract):
        """Per-distribution private submodule init contract for generated CI."""

        _KNOWN_HOSTS_FIELD_COUNT: ClassVar[int] = 3

        known_hosts: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                min_length=1,
                description="Pinned official SSH host-key lines used only in runner temp",
            ),
        ]

        paths: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                min_length=1, description="Submodule paths to init before make setup"
            ),
        ]
        deploy_keys: Annotated[
            tuple[FlextInfraConfigModels.CiPrivateSubmoduleDeployKeySpec, ...],
            m.Field(min_length=1, description="Ordered deploy-key materializations"),
        ]

        @u.model_validator(mode="after")
        def _validate_private_submodule_identity(self) -> Self:
            """Keep path, key, and host identities complete and unambiguous."""
            key_paths = tuple(key.path for key in self.deploy_keys)
            if key_paths != self.paths:
                msg = "private submodule deploy-key paths must exactly match paths"
                raise ValueError(msg)
            for field, values in (
                ("secret", tuple(key.secret for key in self.deploy_keys)),
                ("submodule", tuple(key.submodule for key in self.deploy_keys)),
                ("remote", tuple(key.remote for key in self.deploy_keys)),
                ("known_hosts", self.known_hosts),
            ):
                if len(set(values)) != len(values):
                    msg = f"private submodule {field} values must be unique"
                    raise ValueError(msg)
            for line in self.known_hosts:
                fields = line.split()
                if (
                    len(fields) != self._KNOWN_HOSTS_FIELD_COUNT
                    or fields[0] != "github.com"
                    or fields[1] != "ssh-ed25519"
                ):
                    msg = (
                        "private submodule known_hosts must pin github.com ssh-ed25519"
                    )
                    raise ValueError(msg)
            return self

    class GithubWorkflowRenderSpec(_ConfigContract):
        """Typed input consumed by generated GitHub workflow templates."""

        dist: Annotated[t.NonEmptyStr, m.Field(description="Distribution name")]
        make_profile: Annotated[
            FlextInfraConstantsCodegenProject.MakeProfile,
            m.Field(
                description=(
                    "Make/codegen profile; ci-matrix projected only for "
                    "workspace/standalone; standalone excluded "
                    "and orphan copies pruned"
                )
            ),
        ]
        repository_branch: Annotated[
            t.NonEmptyStr, m.Field(description="Repository integration branch")
        ]
        python_version: Annotated[
            t.NonEmptyStr, m.Field(description="Python major.minor line")
        ]
        dependency_cooldown_days: Annotated[
            int,
            m.Field(
                ge=1, le=90, description="Shared uv and Dependabot dependency cooldown"
            ),
        ]
        github_actions: Annotated[
            Mapping[str, FlextInfraConfigModels.GithubActionPinSpec],
            m.Field(description="Immutable GitHub Action catalog"),
        ]
        make: Annotated[
            FlextInfraConfigModels.MakeSpec,
            m.Field(description="Canonical workflow command contract"),
        ]
        workspace_repositories: Annotated[
            tuple[FlextInfraConfigModels.RepositoryRef, ...],
            m.Field(
                default=(),
                description=(
                    "Governed declared_repository repositories consumed by workspace-scoped "
                    "workflow templates (docs paths, dependabot directories)"
                ),
            ),
        ]
        ci_trigger_branches: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                default=(), description="Ordered, deduplicated blocking CI branches"
            ),
        ] = ()
        has_devcontainer: Annotated[
            bool,
            m.Field(
                default=False,
                description=(
                    "Whether the rendered repository ships a .devcontainer "
                    "directory. Dependabot only accepts a devcontainers "
                    "ecosystem entry when one exists; declaring it for a "
                    "repository without that directory makes GitHub reject the "
                    "whole manifest, which silently disables EVERY ecosystem in "
                    "it, security updates included. Derived from the repository "
                    "on disk rather than declared, because the directory is the "
                    "fact and a second declaration could disagree with it "
                    "(hq-36xk)"
                ),
            ),
        ] = False
        checkout_submodules: Annotated[
            t.NonEmptyStr,
            m.Field(
                default="false",
                pattern=r"^(true|false|recursive)$",
                description=(
                    "actions/checkout submodules mode. Defaults to 'false' "
                    "because the default GITHUB_TOKEN cannot clone sibling "
                    "private repositories: 'recursive' aborts the job at "
                    "checkout with 'Repository not found'. Projects whose "
                    "submodules are public, or that provide a PAT, override "
                    "it per project in codegen.yaml"
                ),
            ),
        ]
        private_submodules: Annotated[
            FlextInfraConfigModels.CiPrivateSubmodulesSpec | None,
            m.Field(
                default=None,
                description=(
                    "Optional private-declared_repository deploy-key init for this "
                    "distribution; None means the workflow skips the step"
                ),
            ),
        ] = None
        system_packages: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                default=(),
                description=(
                    "Runner packages this distribution's tests need; empty "
                    "means the workflow renders no install step"
                ),
            ),
        ] = ()

    class MakeWorkflowRenderSpec(_ConfigContract):
        """Typed input shared by generated local workflow surfaces."""

        dist: Annotated[t.NonEmptyStr, m.Field(description="Distribution name")]
        make: Annotated[
            FlextInfraConfigModels.MakeSpec,
            m.Field(description="Canonical workflow command contract"),
        ]

    class DistroDockerRenderSpec(_ConfigContract):
        """Typed input consumed by generated distro Dockerfiles."""

        package_name: Annotated[
            t.NonEmptyStr, m.Field(description="Python import package name")
        ]
        python_version: Annotated[
            t.NonEmptyStr, m.Field(description="Python major.minor line")
        ]
        make: Annotated[
            FlextInfraConfigModels.MakeSpec,
            m.Field(description="Canonical Make CI token contract for ENV CI=Y"),
        ]

    class UvPackageSelectorSpec(_ConfigContract):
        """Package selector for one official uv scoped dependency exclusion."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Selected package name")]
        version: Annotated[
            t.NonEmptyStr | None,
            m.Field(description="Optional selected package version expression"),
        ] = None

    class UvScopedDependencyExclusionSpec(_ConfigContract):
        """Project-routed official uv scoped dependency exclusion."""

        project: Annotated[
            t.NonEmptyStr,
            m.Field(exclude=True, description="Owning project distribution route"),
        ]
        package: Annotated[
            FlextInfraConfigModels.UvPackageSelectorSpec,
            m.Field(description="Package whose transitive edge is scoped"),
        ]
        dependencies: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(min_length=1, description="Excluded transitive dependency names"),
        ]

    class ProfileSpec(_ConfigContract):
        """Execution semantics for one generated Make profile."""

        name: Annotated[
            FlextInfraConstantsCodegenProject.MakeProfile,
            m.Field(description="Closed Make profile name"),
        ]
        environment_scope: Annotated[
            t.NonEmptyStr, m.Field(description="uv environment ownership")
        ]
        setup_scope: Annotated[
            t.NonEmptyStr, m.Field(description="setup orchestration scope")
        ]
        execution_scope: Annotated[
            t.NonEmptyStr, m.Field(description="check/test runtime scope")
        ]
        discovery_scope: Annotated[
            Literal["gitmodules", "none"],
            m.Field(description="repository-local discovery authority"),
        ]

    class MakeVerbSpec(_ConfigContract):
        """One public Make verb and its complete handler-selector contract."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Public Make verb")]
        default_what: Annotated[
            t.NonEmptyStr, m.Field(description="Default WHAT selector")
        ]
        whats: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(min_length=1, description="Complete ordered handler selectors"),
        ]
        apply_guarded: Annotated[
            bool, m.Field(description="Whether mutation requires APPLY=Y")
        ] = False
        accepts_apply: Annotated[
            bool,
            m.Field(
                description=(
                    "Whether APPLY=Y is legal for this verb without declaring it "
                    "apply-guarded. Used by run handlers that sometimes mutate "
                    "under explicit APPLY but must stay unguarded by default."
                )
            ),
        ] = False
        apply_what: Annotated[
            t.NonEmptyStr,
            m.Field(
                default="all",
                description=(
                    "Selector an apply-guarded or accepts_apply verb resolves "
                    "to when APPLY is set and no explicit WHAT is given. "
                    "Without it, a mutating workflow step could silently "
                    "retain its read-only default selector"
                ),
            ),
        ]

        @u.model_validator(mode="after")
        def _validate_whats(self) -> Self:
            """Require defaults and mutation selectors to name one owned handler."""
            if len(set(self.whats)) != len(self.whats):
                msg = f"make verb {self.name} handler selectors must be unique"
                raise ValueError(msg)
            required = {self.default_what}
            if self.apply_guarded or self.accepts_apply:
                required.add(self.apply_what)
            if self.apply_guarded and self.accepts_apply:
                msg = (
                    f"make verb {self.name} cannot set both apply_guarded and "
                    "accepts_apply"
                )
                raise ValueError(msg)
            if self.apply_guarded and self.default_what == self.apply_what:
                msg = (
                    f"make verb {self.name} apply_guarded default_what must "
                    "differ from apply_what so the fixed-point re-check "
                    "runs without APPLY=Y"
                )
                raise ValueError(msg)
            missing = required - set(self.whats)
            if missing:
                msg = (
                    f"make verb {self.name} selectors have no handler: "
                    f"{', '.join(sorted(missing))}"
                )
                raise ValueError(msg)
            return self

    class MakeWorkflowStepSpec(_ConfigContract):
        """One canonical workflow step and its explicit mutation intent."""

        verb: Annotated[t.NonEmptyStr, m.Field(description="Declared public verb")]
        apply: Annotated[
            bool,
            m.Field(description="Whether the step supplies the configured apply token"),
        ] = False
        contexts: Annotated[
            tuple[Literal["local", "ci", "pre_commit", "pre_push"], ...],
            m.Field(
                min_length=1,
                description="Execution contexts consuming this single workflow row",
            ),
        ]
        what: Annotated[
            t.NonEmptyStr | None,
            m.Field(
                default=None,
                description=(
                    "Explicit WHAT selector for this step. None lets the verb "
                    "resolve its own default_what, so a row names a selector "
                    "only when it deliberately departs from that default."
                ),
            ),
        ] = None
        gates_skip: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                default=(),
                description=(
                    "Gate ids this step omits when it runs from a hook context "
                    "(pre_commit/pre_push). Local and CI invocations of the "
                    "same verb keep the full default set."
                ),
            ),
        ] = ()

        @u.model_validator(mode="after")
        def _validate_contexts(self) -> Self:
            """Require unique contexts and retain every step in the local workflow."""
            if len(set(self.contexts)) != len(self.contexts):
                msg = f"make workflow contexts must be unique for {self.verb}"
                raise ValueError(msg)
            if "local" not in self.contexts:
                msg = f"make workflow step {self.verb} must run locally"
                raise ValueError(msg)
            allowed = set(FlextInfraConstantsMake.PROJECT_CHECK_GATES_ALLOWED_VALUES)
            unknown = sorted(set(self.gates_skip) - allowed)
            if unknown:
                msg = (
                    f"make workflow step {self.verb} gates_skip contains "
                    f"unknown gates: {', '.join(unknown)}"
                )
                raise ValueError(msg)
            return self

    class MakeCiSpec(_ConfigContract):
        """The only permitted environment delta between local and CI execution."""

        variable: Annotated[t.NonEmptyStr, m.Field(description="CI environment key")]
        value: Annotated[t.NonEmptyStr, m.Field(description="CI environment value")]
        local_value: Annotated[
            t.NonEmptyStr,
            m.Field(
                description=(
                    "Local form of the CI ternary. A hook declares this value "
                    "explicitly so an inherited CI token from the caller can "
                    "never revoke pytest or the type-checker gates."
                )
            ),
        ] = "N"
        local_check_gates: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                description=(
                    "Gate ids run by make check under the local CI token: the "
                    "slow whole-program type checkers. This is the ONLY "
                    "declared set; the CI token runs its strict complement and "
                    "an unset token runs every allowed gate."
                )
            ),
        ] = FlextInfraConstantsMake.PROJECT_CHECK_GATES_LOCAL_VALUES

        @u.model_validator(mode="after")
        def _validate_local_check_gates(self) -> Self:
            """Every locally owned gate must be in the allowed check vocabulary."""
            allowed = set(FlextInfraConstantsMake.PROJECT_CHECK_GATES_ALLOWED_VALUES)
            unknown = sorted(set(self.local_check_gates) - allowed)
            if unknown:
                msg = (
                    "make.ci.local_check_gates contains unknown gates: "
                    f"{', '.join(unknown)}"
                )
                raise ValueError(msg)
            return self

        @m.computed_field
        @property
        def check_gates(self) -> tuple[str, ...]:
            """Gates run under the CI token, as the strict complement.

            CI=Y is the inverse of CI=N by construction, never a second list: a
            gate that moves into or out of ``local_check_gates`` moves out of or
            into this set in the same edit, so the two can never overlap nor
            leave a gate unowned.

            The complement is taken over the DEFAULT vocabulary, not ALLOWED.
            ``format`` is allowed as an explicit ``CHECK_GATES=format`` request
            but is absent from the default set because it MUTATES files, so
            deriving over ALLOWED silently scheduled a formatter inside CI's
            read-only check — the one thing the comment on ``local_check_gates``
            says must never happen.
            """
            local = frozenset(self.local_check_gates)
            return tuple(
                gate
                for gate in FlextInfraConstantsMake.PROJECT_CHECK_GATES_DEFAULT_VALUES
                if gate not in local
            )

    class ScriptDispatchSpec(_ConfigContract):
        """Opt-in routing of non-builtin verbs to a script command framework."""

        dispatcher: Annotated[
            t.NonEmptyStr,
            m.Field(
                description=(
                    "Repository-relative dispatcher entrypoint that resolves "
                    "scripts/<verb>/<what>.{py,sh} commands"
                )
            ),
        ]
        roots: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                min_length=1,
                description=(
                    "Repository-relative script roots scanned for a matching "
                    "<verb>/<what> command before falling back to a builtin"
                ),
            ),
        ]

    class CustomHandlerPolicy(_ConfigContract):
        """Strict schema for the only handwritten Make extension file."""

        filename: Annotated[
            t.NonEmptyStr, m.Field(description="Versioned custom handler filename")
        ]
        target_pattern: Annotated[
            t.NonEmptyStr,
            m.Field(description="Required private target regular expression"),
        ]
        allow_public_targets: bool = m.Field(description="Permit public targets")
        allow_toolchain_declarations: bool = m.Field(
            description="Permit toolchain declarations"
        )

    class CustomHandlerPolicyOverride(_ConfigContract):
        """Per-profile relaxation of the strict custom-handler contract.

        Every field is optional: a profile declares ONLY what it relaxes, so a
        new permission added to the base policy propagates automatically
        instead of having to be repeated in each profile.
        """

        allow_public_targets: bool | None = m.Field(
            default=None, description="Permit public targets"
        )
        allow_toolchain_declarations: bool | None = m.Field(
            default=None, description="Permit toolchain declarations"
        )

    class DocsGithubRepoSpec(_ConfigContract):
        """One governed GitHub repository used for cross-repo doc links."""

        organization: Annotated[
            t.NonEmptyStr, m.Field(description="GitHub organization")
        ]
        repository: Annotated[t.NonEmptyStr, m.Field(description="GitHub repository")]
        branch: Annotated[
            t.NonEmptyStr, m.Field(description="Working-line branch for doc links")
        ]
        local_checkout: Annotated[
            str,
            m.Field(
                default="",
                description=(
                    "Optional local checkout path (~ expanded) for existence checks"
                ),
            ),
        ] = ""

    class MakeCleanSpec(_ConfigContract):
        """Disposable artifacts the generated clean verb removes.

        Stale caches and traces cause FALSE DIAGNOSES, so the disposable set is
        declared data rather than a literal buried in a recipe: every project
        cleans exactly the same things and a new artifact kind is one config row.
        """

        cache_dirs: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(description="Cache directory names removed anywhere in the tree"),
        ]
        root_dirs: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(description="Directories removed at the project root only"),
        ]
        root_files: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(description="Files removed at the project root only"),
        ]
        trace_globs: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(description="Trace/profile globs removed anywhere in the tree"),
        ]

    class MakeDocsSpec(_ConfigContract):
        """Generated Makefile docs verb lifecycle and audit policy."""

        api_modules: Annotated[
            Mapping[t.NonEmptyStr, tuple[t.NonEmptyStr, ...]],
            m.Field(
                min_length=1,
                description=(
                    "Public API modules generated per distribution; absent "
                    "distributions own no module pages"
                ),
            ),
        ]
        mutable_actions: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(min_length=1, description="Docs actions guarded by APPLY=Y"),
        ]
        reports_dir: Annotated[
            Path, m.Field(description="Repository-relative docs reports directory")
        ]
        cross_project_relative_link_pattern: Annotated[
            t.NonEmptyStr,
            m.Field(
                description="Regex rejecting cross-project relative Markdown links"
            ),
        ]
        stale_github_organizations: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                default=("organization",),
                description="Placeholder GitHub orgs that must be rewritten",
            ),
        ] = ("organization",)
        github_repos: Annotated[
            tuple[FlextInfraConfigModels.DocsGithubRepoSpec, ...],
            m.Field(
                default=(),
                description="Governed org/repo/branch map for cross-repo doc URLs",
            ),
        ] = ()

        @u.model_validator(mode="after")
        def _validate_api_modules(self) -> Self:
            """Reject duplicate or non-importable API module declarations."""
            for distribution, modules in self.api_modules.items():
                if not modules:
                    msg = f"docs api_modules must not be empty: {distribution}"
                    raise ValueError(msg)
                if len(set(modules)) != len(modules):
                    msg = f"docs api_modules must be unique: {distribution}"
                    raise ValueError(msg)
                invalid = next(
                    (
                        module
                        for module in modules
                        if not all(part.isidentifier() for part in module.split("."))
                    ),
                    None,
                )
                if invalid is not None:
                    msg = f"docs api module is not importable: {invalid}"
                    raise ValueError(msg)
            return self

    class TestmonCacheSpec(_ConfigContract):
        """Adaptive pytest-testmon GitHub Actions cache policy."""

        schema_version: Annotated[
            int, m.Field(ge=1, description="Cache key schema version")
        ]
        mode: Annotated[
            Literal["bootstrap", "stable"], m.Field(description="Cache renewal phase")
        ]
        save_enabled: Annotated[
            bool,
            m.Field(
                description=(
                    "Whether CI may upload a new testmon generation. False while "
                    "quota/HTTP 402 blocks fleet-wide saves (QUOTA_HOLD)."
                )
            ),
        ]
        max_bootstrap_generations: Annotated[
            int, m.Field(ge=1, le=10, description="Max retained bootstrap generations")
        ]
        max_stable_generations: Annotated[
            int, m.Field(ge=1, le=10, description="Max retained stable generations")
        ]
        per_repo_budget_bytes: Annotated[
            int, m.Field(ge=1, description="Per-repo testmon namespace budget in bytes")
        ]
        warning_threshold_percent: Annotated[
            int, m.Field(ge=1, le=100, description="Quota warning threshold percent")
        ]
        maintenance_threshold_percent: Annotated[
            int,
            m.Field(ge=1, le=100, description="Quota maintenance threshold percent"),
        ]
        block_threshold_percent: Annotated[
            int, m.Field(ge=1, le=100, description="Quota block-save threshold percent")
        ]
        allowed_save_refs: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(min_length=1, description="Refs allowed to save cache generations"),
        ]
        key_prefix: Annotated[
            t.NonEmptyStr, m.Field(description="Immutable cache key prefix")
        ]

        @u.model_validator(mode="after")
        def _validate_thresholds(self) -> Self:
            """Require warning < maintenance < block."""
            if not (
                self.warning_threshold_percent
                < self.maintenance_threshold_percent
                < self.block_threshold_percent
            ):
                msg = "testmon cache thresholds must satisfy warning < maintenance < block"
                raise ValueError(msg)
            return self

    class MakeWorkInProgressSpec(_ConfigContract):
        """Predicate for work-in-progress branches and draft-PR gate behavior.

        A hook that runs the full gate matrix on every push turns an
        in-progress branch into a stop-and-wait loop, so contributors start
        bypassing the hook entirely -- which costs more than it saves. The
        predicate is DATA so the escape is declared and auditable rather than
        improvised per-repository with `--no-verify`.
        """

        draft_pr: Annotated[
            bool, m.Field(description="Treat GitHub draft PRs as work-in-progress")
        ]
        branch_patterns: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                min_length=1,
                description="Regex patterns that mark a branch as work-in-progress",
            ),
        ]
        merge_lock_target_branches: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                min_length=1,
                description="Target branches that are blocked for WIP merges",
            ),
        ]

    class MakeSpec(_ConfigContract):
        """Complete generated Makefile public and extension contract."""

        work_in_progress: Annotated[
            FlextInfraConfigModels.MakeWorkInProgressSpec,
            m.Field(description="WIP branch and draft PR gate predicate"),
        ]
        selector: Annotated[
            t.NonEmptyStr, m.Field(description="Single selector variable name")
        ]
        apply_variable: Annotated[
            t.NonEmptyStr, m.Field(description="Write-enable variable name")
        ]
        apply_value: Annotated[
            t.NonEmptyStr, m.Field(description="Only accepted write-enable value")
        ]
        apply_absent_value: Annotated[
            t.NonEmptyStr,
            m.Field(
                default="N",
                description=(
                    "Value the generated Makefile seeds when the caller enables "
                    "nothing. It is forwarded verbatim on every read-only run, "
                    "so the boundary must read it as 'not applying' instead of "
                    "as an invalid write-enable token"
                ),
            ),
        ]
        # Why (operator law 2026-08-24): git-hook stages are OFF by default and
        # re-enabled case by case via these config gates. The workflow keeps
        # owning WHICH steps belong to each stage; the booleans only govern
        # whether the stage is generated and installed at all.
        pre_commit: Annotated[
            bool,
            m.Field(
                default=False,
                description="Generate and install the pre-commit git-hook stage",
            ),
        ] = False
        pre_push: Annotated[
            bool,
            m.Field(
                default=False,
                description="Generate and install the pre-push git-hook stage",
            ),
        ] = False
        workflow: Annotated[
            tuple[FlextInfraConfigModels.MakeWorkflowStepSpec, ...],
            m.Field(min_length=1, description="Ordered canonical validation workflow"),
        ]
        ci: Annotated[
            FlextInfraConfigModels.MakeCiSpec,
            m.Field(description="Config-owned CI-only environment delta"),
        ]
        testmon_cache: Annotated[
            FlextInfraConfigModels.TestmonCacheSpec,
            m.Field(description="Adaptive testmon Actions cache policy"),
        ]
        verbs: Annotated[
            tuple[FlextInfraConfigModels.MakeVerbSpec, ...],
            m.Field(description="Ordered canonical public verbs"),
        ]
        clean: Annotated[
            FlextInfraConfigModels.MakeCleanSpec,
            m.Field(description="Disposable artifacts removed by the clean verb"),
        ]
        docs: Annotated[
            FlextInfraConfigModels.MakeDocsSpec,
            m.Field(description="Public documentation lifecycle policy"),
        ]
        custom_handler_policy: Annotated[
            FlextInfraConfigModels.CustomHandlerPolicy,
            m.Field(description="Private custom target policy"),
        ]
        custom_handler_profile_overrides: Annotated[
            Mapping[t.NonEmptyStr, FlextInfraConfigModels.CustomHandlerPolicyOverride],
            m.Field(
                default_factory=immutable_empty_mapping,
                description="Per-profile overrides of the custom handler policy",
            ),
        ]
        project_check_gates: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                default=(),
                description=(
                    "Check gates this project implements itself, unioned into "
                    "the built-in vocabulary. Without this seam the vocabulary "
                    "is a closed Final tuple, so a project that ships a working "
                    "`_custom_check_<gate>` handler still cannot reach it "
                    "through `make check`: the gate is rejected as unknown. A "
                    "gate that can never run is not a gate, which is how "
                    "references, agents, census and waza ended up outside the "
                    "gate matrix in consuming repositories. Each id must have a "
                    "`_custom_check_<id>` handler in the project's "
                    f"{FlextInfraConstantsCodegenProject.CUSTOM_MAKE_FILENAME}."
                ),
            ),
        ] = ()

        @u.model_validator(mode="after")
        def _validate_project_check_gates(self) -> Self:
            """Project gates must be unique and must not shadow a built-in."""
            if len(set(self.project_check_gates)) != len(self.project_check_gates):
                msg = "make project_check_gates must be unique"
                raise ValueError(msg)
            builtin = set(FlextInfraConstantsMake.PROJECT_CHECK_GATES_ALLOWED_VALUES)
            shadowed = sorted(set(self.project_check_gates) & builtin)
            if shadowed:
                msg = (
                    "make project_check_gates shadow built-in gates: "
                    f"{', '.join(shadowed)}"
                )
                raise ValueError(msg)
            return self

        @u.model_validator(mode="after")
        def _validate_verbs(self) -> Self:
            """Validate declared public verbs against workflow and contract.

            Why there is no `"setup" in declared` rejection here (flext-lq86m):
            the message it carried -- "make setup cannot require the managed
            validation environment" -- is a statement about a verb's
            ENVIRONMENT DEPENDENCY, not about the name `setup`. Testing the
            name was the wrong predicate, and it contradicted
            `config/codegen.yaml`, which declares `setup` as a public verb; the
            config singleton builds eagerly at import, so every entrypoint died
            on that contradiction. The real invariant is enforced structurally
            in the template, which excludes `setup` from
            `_builtin_require_environment` (`$(filter-out setup,$(PUBLIC_VERBS))`
            and `{% raw %}{% for verb in make.verbs if verb.name != "setup" %}{% endraw %}`),
            so `setup` never depends on the environment it exists to create.
            """
            declared = {verb.name for verb in self.verbs}
            if len(declared) != len(self.verbs):
                msg = "make public verb names must be unique"
                raise ValueError(msg)
            # Why (hq-36xk, flext-lq86m): the guard that lived here read
            # `if "setup" in serialized` and protected `make setup` from being
            # placed in the serialized mutation set, so it could never require
            # the managed validation environment it is supposed to CREATE.
            # 3e5fbc747 exterminated the serialize-make lifecycle and deleted
            # `self.serialization`, but rebound this condition to `declared`
            # instead of removing it with the concept it guarded. `setup` is a
            # mandatory canonical verb, so the inverted check rejected every
            # valid configuration and `flext_infra.config` could not be built at
            # all. The serialized set no longer exists; the guard has no object.
            workflow_verbs = tuple(step.verb for step in self.workflow)
            if len(set(workflow_verbs)) != len(workflow_verbs):
                msg = "make workflow verbs must be unique"
                raise ValueError(msg)
            unknown_workflow = set(workflow_verbs) - declared
            if unknown_workflow:
                msg = (
                    "make workflow verbs are not declared public verbs: "
                    f"{', '.join(sorted(unknown_workflow))}"
                )
                raise ValueError(msg)
            verb_specs = {verb.name: verb for verb in self.verbs}
            invalid_apply = [
                step.verb
                for step in self.workflow
                if step.apply != verb_specs[step.verb].apply_guarded
            ]
            if invalid_apply:
                msg = (
                    "make workflow apply intent must match verb contract: "
                    f"{', '.join(sorted(invalid_apply))}"
                )
                raise ValueError(msg)
            docs_verb = next((verb for verb in self.verbs if verb.name == "docs"), None)
            if docs_verb is None:
                msg = "make docs verb must be declared"
                raise ValueError(msg)
            docs_actions = set(docs_verb.whats)
            invalid_mutable = set(self.docs.mutable_actions) - docs_actions
            if invalid_mutable:
                msg = "make docs mutable_actions must be declared in actions"
                raise ValueError(msg)
            if (
                self.docs.reports_dir.is_absolute()
                or ".." in self.docs.reports_dir.parts
            ):
                msg = "make docs reports_dir must be repository-relative"
                raise ValueError(msg)
            return self

        @m.computed_field
        @property
        def handler_whats(self) -> Mapping[str, tuple[str, ...]]:
            """Canonical public verb-to-handler matrix consumed by every renderer.

            The check verb's what-selectors are augmented with the full gate
            vocabulary (check_gates_allowed) so a consumer can address an
            individual gate with `make check WHAT=<gate>`. The gates are
            rendered into `_ALLOWED_WHATS_check` by Makefile.j2, and per-gate
            `_builtin_check_<gate>` handlers are generated to back them.
            """
            whats: dict[str, tuple[str, ...]] = {
                verb.name: verb.whats for verb in self.verbs
            }
            # Why (flext-7b5x9): a project declaring its own gates via
            # project_check_gates (and every built-in gate) must be reachable
            # through the dispatch validator. Without this augmentation the
            # check verb's `whats` stays `[all]`, so `make check WHAT=lint`
            # is rejected as "unsupported" even though CHECK_GATES_ALLOWED
            # would accept the gate value itself -- the dispatch gate is
            # strictly narrower than the runtime gate set it claims.
            if "check" in whats:
                whats["check"] = tuple(
                    dict.fromkeys((*whats["check"], *self.check_gates_allowed))
                )
            return whats

        @m.computed_field
        @property
        def check_gates_allowed(self) -> tuple[str, ...]:
            """Canonical generated Make check-gate vocabulary.

            The built-in gates this package implements, plus the gates the
            project declares for itself. The built-in tuple is the BASE, never
            the whole vocabulary: a consuming repository owns gates this
            package knows nothing about, and rejecting them as unknown is what
            kept working handlers unreachable from `make check`.
            """
            return (
                *FlextInfraConstantsMake.PROJECT_CHECK_GATES_ALLOWED_VALUES,
                *self.project_check_gates,
            )

        @m.computed_field
        @property
        def check_gates_default(self) -> tuple[str, ...]:
            """Canonical generated Make default check gates.

            A declared project gate runs by default, exactly like a built-in:
            a gate that must be asked for by name is a gate nobody runs.
            """
            return (
                *FlextInfraConstantsMake.PROJECT_CHECK_GATES_DEFAULT_VALUES,
                *self.project_check_gates,
            )

        @m.computed_field
        @property
        def check_gates_fixable(self) -> tuple[str, ...]:
            """Gates ``make fix APPLY=Y`` can actually repair.

            Asking for a gate that cannot fix anything still pays its full cost;
            a fix pass built from the ALLOWED vocabulary once timed out doing
            exactly that.
            """
            return FlextInfraConstantsMake.PROJECT_CHECK_GATES_FIXABLE_VALUES

        @m.computed_field
        @property
        def custom_handler_policies(
            self,
        ) -> Mapping[str, FlextInfraConfigModels.CustomHandlerPolicy]:
            """Effective custom-handler policy for every Make profile.

            The base policy states the strictest contract (private handlers
            only). A profile whose custom surface legitimately owns more --
            a repository root orchestrating its declared_repositories -- declares only the
            fields it relaxes, so the engine never has to know which project
            it is conforming.
            """
            base = self.custom_handler_policy
            overrides = self.custom_handler_profile_overrides
            # Keys are normalised to the profile's string value: MakeProfile is a
            # StrEnum, so a raw YAML key and its enum declared_repository must land on the SAME
            # entry. Mixing both would make a lookup silently miss and fall back to
            # the strict base policy.
            return {
                str(profile): (
                    base.model_copy(update=override.model_dump(exclude_none=True))
                    if (override := overrides.get(str(profile)))
                    else base
                )
                for profile in (
                    *overrides,
                    *FlextInfraConstantsCodegenProject.MakeProfile,
                )
            }

    class ManagedFileSpec(_ConfigContract):
        """One versioned file governed by codegen lifecycle policy."""

        path: Annotated[Path, m.Field(description="Repository-relative file path")]
        owner: Annotated[t.NonEmptyStr, m.Field(description="Canonical owner")]
        policy: Annotated[
            Literal["full", "merge", "create-only", "delegated", "manual"],
            m.Field(
                description=(
                    "Conform ownership and mutation policy; create-only files are "
                    "emitted during creation, preserved when present, and never "
                    "backfilled into existing trees"
                )
            ),
        ]
        executable: Annotated[
            bool | None,
            m.Field(description="Required executable state when mode is governed"),
        ] = None
        conflict_sections: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                description=(
                    "Dotted sections the owner renders, so a merge conflict in "
                    "them is resolvable by re-rendering rather than by hand. A "
                    "section the owner produces but does not declare here "
                    "dead-ends the merge: absorbing an integration base that "
                    "still carries the previous projection leaves a conflict "
                    "the canonical surface cannot resolve."
                )
            ),
        ] = ()

    class TemplateEntrySpec(_ConfigContract):
        """One scaffold-only template mapping consumed by ``codegen new``."""

        source: Annotated[Path, m.Field(description="Template-root-relative source")]
        destination: Annotated[
            t.NonEmptyStr,
            m.Field(description="Tokenized repository-relative destination"),
        ]
        profiles: Annotated[
            tuple[FlextInfraConstantsCodegenProject.MakeProfile, ...],
            m.Field(description="Profiles that consume the template"),
        ]
        delegate: Annotated[
            t.NonEmptyStr, m.Field(description="Canonical rendering delegate")
        ]
        overwrite: Annotated[
            bool, m.Field(description="Whether the template owns existing content")
        ] = False

    class TemplatesSpec(_ConfigContract):
        """New-project scaffold root and its complete ordered manifest."""

        root: Annotated[Path, m.Field(description="Package-relative template root")]
        entries: Annotated[
            tuple[FlextInfraConfigModels.TemplateEntrySpec, ...],
            m.Field(description="Complete ordered template manifest"),
        ]

    class ScaffoldBuildSpec(_ConfigContract):
        """Configured Python build backend for newly scaffolded projects."""

        backend: Annotated[t.NonEmptyStr, m.Field(description="PEP 517 backend")]
        requirements: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(min_length=1, description="Build-system requirements"),
        ]

    class ScaffoldDependencyProfileSpec(_ConfigContract):
        """Dependencies selected by the declared upstream FLEXT facade."""

        upstream: Annotated[
            t.NonEmptyStr, m.Field(description="Supported upstream facade package")
        ]
        runtime: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(min_length=1, description="Runtime requirements"),
        ]
        codegen: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(description="Code-generation requirements"),
        ] = ()

    class ScaffoldProjectSpec(_ConfigContract):
        """Project metadata policy for newly scaffolded distributions."""

        readme: Annotated[t.NonEmptyStr, m.Field(description="PEP 621 readme path")]
        supported_licenses: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(min_length=1, description="Licenses with complete templates"),
        ]
        classifiers: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(min_length=1, description="Default PyPI classifiers"),
        ]
        keywords: Annotated[
            tuple[t.NonEmptyStr, ...], m.Field(description="Default project keywords")
        ] = ()
        dev: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                min_length=1,
                description="Canonical development and validation requirements",
            ),
        ]
        dependency_profiles: Annotated[
            tuple[FlextInfraConfigModels.ScaffoldDependencyProfileSpec, ...],
            m.Field(min_length=1, description="Upstream dependency profiles"),
        ]

    class ScaffoldPingExampleSpec(_ConfigContract):
        """Values for the functional ping example created only by codegen new."""

        command_name: Annotated[
            t.NonEmptyStr, m.Field(description="Public CLI command")
        ]
        help_text: Annotated[
            t.NonEmptyStr, m.Field(description="Public CLI command help")
        ]
        success_message: Annotated[
            t.NonEmptyStr, m.Field(description="CLI success message")
        ]
        enabled_default: Annotated[
            bool, m.Field(description="Default runtime enablement")
        ]
        reply: Annotated[t.NonEmptyStr, m.Field(description="Enabled ping response")]
        disabled_reply: Annotated[
            t.NonEmptyStr, m.Field(description="Disabled ping response")
        ]

    class ScaffoldGitignoreSectionSpec(_ConfigContract):
        """One configured section of the generated Git ignore policy."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Section heading")]
        patterns: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(min_length=1, description="Ignored path patterns"),
        ]
        profiles: Annotated[
            tuple[FlextInfraConstantsCodegenProject.MakeProfile, ...],
            m.Field(
                description=(
                    "Make profiles this section applies to; empty means every "
                    "profile (universal). Sections that only make sense at the "
                    "superproject root (declared_repository-directory allowlists, workspace "
                    "manifest, submodule/Beads coordination) declare "
                    "[workspace] so declared_repositories and standalone projects never "
                    "receive the phantom entries."
                )
            ),
        ] = ()

    class ScaffoldSpec(_ConfigContract):
        """Complete typed policy consumed only by new-project templates."""

        build: Annotated[
            FlextInfraConfigModels.ScaffoldBuildSpec,
            m.Field(description="Build-system policy"),
        ]
        project: Annotated[
            FlextInfraConfigModels.ScaffoldProjectSpec,
            m.Field(description="Project metadata and dependency policy"),
        ]
        ping_example: Annotated[
            FlextInfraConfigModels.ScaffoldPingExampleSpec,
            m.Field(description="Functional scaffold example"),
        ]
        gitignore_sections: Annotated[
            tuple[FlextInfraConfigModels.ScaffoldGitignoreSectionSpec, ...],
            m.Field(min_length=1, description="Generated Git ignore sections"),
        ]

    class GitignoreRenderContext(_ConfigContract):
        """Profile-filtered input consumed by the Git ignore template."""

        gitignore_sections: Annotated[
            tuple[FlextInfraConfigModels.ScaffoldGitignoreSectionSpec, ...],
            m.Field(min_length=1, description="Applicable Git ignore sections"),
        ]

    class RepositoryRef(_ConfigContract):
        """One declared repository and its immutable Git origin contract."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(use_enum_values=False)

        name: Annotated[t.NonEmptyStr, m.Field(description="Catalog key")]
        distribution: Annotated[
            t.NonEmptyStr, m.Field(description="Python distribution or repository name")
        ]
        url: Annotated[
            t.NonEmptyStr,
            m.Field(description="Canonical GitHub clone URL ending in .git"),
        ]
        path: Annotated[
            Path, m.Field(description="POSIX path relative to its repository root")
        ]
        role: Annotated[
            FlextInfraConstantsCodegenProject.MakeProfile,
            m.Field(description="Repository role in the declared topology"),
        ]
        state: Annotated[
            FlextInfraConstantsCodegenProject.RepositoryState,
            m.Field(description="Repository lifecycle state"),
        ] = FlextInfraConstantsCodegenProject.RepositoryState.ACTIVE
        provider: Annotated[
            t.NonEmptyStr,
            m.Field(description="Provider key from the codegen configuration"),
        ]
        checkout: Annotated[
            FlextInfraConstantsCodegenProject.CheckoutKind,
            m.Field(description="Physical checkout topology"),
        ]
        codegen: Annotated[
            FlextInfraConstantsCodegenProject.CodegenKind,
            m.Field(description="Repository code-generation policy"),
        ]
        package: Annotated[
            bool, m.Field(description="Repository publishes a Python package")
        ]
        editable: Annotated[
            bool, m.Field(description="Overlay repository as an editable dependency")
        ]
        read_only: Annotated[
            bool, m.Field(description="Repository rejects generated mutations")
        ]
        uv_link_mode: Annotated[
            Literal["clone", "copy", "hardlink", "symlink"] | None,
            m.Field(
                description=(
                    "Repository-specific uv installation link mode; absent uses "
                    "the fleet toolchain default"
                )
            ),
        ] = None
        dependency_cooldown_exclusions: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                description=(
                    "Repository-scoped packages explicitly exempted from the "
                    "fleet dependency cooldown"
                )
            ),
        ] = ()
        dependency_cooldown_overrides: Annotated[
            t.StrMapping,
            m.Field(
                default_factory=immutable_empty_mapping,
                description=(
                    "Repository-scoped package cutoffs projected to uv "
                    "exclude-newer-package"
                ),
            ),
        ]
        extra_verbs: Annotated[
            tuple[FlextInfraConfigModels.MakeVerbSpec, ...],
            m.Field(
                description=(
                    "Additional public Make verbs this repository dispatches "
                    "beyond the canonical set (e.g. a script command framework)"
                )
            ),
        ] = ()
        script_dispatch: Annotated[
            FlextInfraConfigModels.ScriptDispatchSpec | None,
            m.Field(
                description=(
                    "Opt-in script command-framework routing for non-builtin "
                    "verbs and WHAT selectors; None keeps builtin-only dispatch"
                )
            ),
        ] = None

        @u.model_validator(mode="after")
        def _validate_dependency_cooldown_policy(self) -> Self:
            """Reject duplicate or contradictory repository cooldown entries."""
            if len(set(self.dependency_cooldown_exclusions)) != len(
                self.dependency_cooldown_exclusions
            ):
                msg = "repository dependency cooldown exclusions must be unique"
                raise ValueError(msg)
            overlap = set(self.dependency_cooldown_exclusions).intersection(
                self.dependency_cooldown_overrides
            )
            if overlap:
                msg = (
                    "repository dependency cooldown package cannot be both excluded "
                    f"and overridden: {', '.join(sorted(overlap))}"
                )
                raise ValueError(msg)
            return self

        @u.field_serializer("dependency_cooldown_overrides", when_used="json")
        def _serialize_dependency_cooldown_overrides(
            self, value: t.StrMapping
        ) -> dict[str, str]:
            """Project the immutable mapping through JSON/template boundaries."""
            return dict(value)

    class BeadsProjectSpec(_ConfigContract):
        """Repository-local Beads identity from ``config/beads.yaml``."""

        version: Annotated[
            Literal[1],
            m.Field(description="Beads project configuration schema version"),
        ]
        workspace: Annotated[
            t.NonEmptyStr, m.Field(description="Stable workspace identity")
        ]
        database: Annotated[
            t.NonEmptyStr, m.Field(description="Repository-owned Dolt database")
        ]
        issue_prefix: Annotated[
            t.NonEmptyStr, m.Field(description="Repository-owned issue prefix")
        ]
        custom_issue_types: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                description="Repository-owned custom types beyond the Gas City baseline"
            ),
        ] = ()

        @u.model_validator(mode="after")
        def _validate_custom_issue_types(self) -> Self:
            """Reject duplicate project extensions before projection."""
            if len(set(self.custom_issue_types)) != len(self.custom_issue_types):
                msg = "beads custom_issue_types must be unique"
                raise ValueError(msg)
            return self

    class RepositoryConformTarget(_ConfigContract):
        """Runtime-derived conformance identity for one repository."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(use_enum_values=False)

        repository: Annotated[
            FlextInfraConfigModels.RepositoryRef,
            m.Field(description="Declared immutable repository identity"),
        ]
        root: Annotated[
            Path, m.Field(description="Resolved repository root receiving conformance")
        ]
        make_profile: Annotated[
            FlextInfraConstantsCodegenProject.MakeProfile,
            m.Field(description="Make profile inferred from live Git topology"),
        ]
        beads: Annotated[
            FlextInfraConfigModels.BeadsProjectSpec | None,
            m.Field(
                description=(
                    "Repository-local Beads identity. Absent on a target derived "
                    "from declarations only: that route owns no ledger projection "
                    "and must never read the .beads tree to invent one."
                )
            ),
        ] = None
        canonical_project_name: Annotated[
            t.NonEmptyStr, m.Field(description="Canonical PEP 621 project name")
        ]
        baseline_branch: Annotated[
            t.NonEmptyStr,
            m.Field(description="Provider-owned integration ancestry baseline"),
        ]
        ci_enabled: Annotated[
            bool, m.Field(description="Whether conform owns the CI projection")
        ]
        external_dependency_paths: Annotated[
            tuple[Path, ...],
            m.Field(description="Observed external or fork Git submodule paths"),
        ] = ()
        technical_branch_patterns: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(description="Technical branches excluded from ancestry policy"),
        ] = ()
        governed_branch_patterns: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                min_length=1,
                description=(
                    "Development lines gated by ancestry policy; required because "
                    "an empty tuple would match no ref and silently disable the "
                    "gate instead of failing closed"
                ),
            ),
        ]

    class ManagedGitlinkSpec(_ConfigContract):
        """One governed submodule with its provider-owned baseline branch."""

        repository: Annotated[
            FlextInfraConfigModels.RepositoryRef,
            m.Field(description="Governed repository identity"),
        ]
        branch: Annotated[
            t.NonEmptyStr,
            m.Field(description="Declared gitlink branch (. follows the superproject)"),
        ]

    class MakeCommandContext(_ConfigContract):
        """Shared command identity required by every generated Make surface."""

        infra_cli: Annotated[
            t.NonEmptyStr, m.Field(description="Installed infrastructure CLI command")
        ]
        pytest: Annotated[
            FlextInfraModelsDepsToolSettings.PytestConfig,
            m.Field(description="Typed pytest execution policy"),
        ]

    class MakefileRenderSpec(MakeCommandContext):
        """Field-only render input for an existing repository Makefile."""

        dist: Annotated[t.NonEmptyStr, m.Field(description="PEP 621 project name")]
        make_profile: Annotated[
            FlextInfraConstantsCodegenProject.MakeProfile,
            m.Field(description="Selected repository Make profile"),
        ]
        repository_root_rel: Annotated[
            t.NonEmptyStr, m.Field(description="Relative repository root path")
        ]
        declared_repositories: Annotated[
            tuple[str, ...],
            m.Field(description="Declared workspace declared_repository paths"),
        ] = ()
        workspace_repositories: Annotated[
            tuple[FlextInfraConfigModels.RepositoryRef, ...],
            m.Field(description="Repositories editable from the selected workspace"),
        ] = ()
        workspace_gitlinks: Annotated[
            tuple[FlextInfraConfigModels.ManagedGitlinkSpec, ...],
            m.Field(description="Provider-resolved governed Git submodules"),
        ] = ()
        uv_link_mode: Annotated[
            t.NonEmptyStr, m.Field(description="Configured uv installation link mode")
        ]
        uv_version: Annotated[
            t.NonEmptyStr,
            m.Field(description="mise-owned uv version used by bootstrap validation"),
        ]
        uv_exclude_newer: Annotated[
            t.NonEmptyStr,
            m.Field(description="uv exclude-newer cooldown window for [tool.uv]"),
        ]
        dependency_cooldown_exclusions: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(description="Packages exempted from uv dependency cooldown"),
        ] = ()
        dependency_cooldown_overrides: Annotated[
            t.StrMapping,
            m.Field(
                default_factory=immutable_empty_mapping,
                description="Per-package cooldown cutoffs as RFC 3339 timestamps",
            ),
        ]
        make: Annotated[
            FlextInfraConfigModels.MakeSpec,
            m.Field(description="Generated Make command contract"),
        ]
        extra_verbs: Annotated[
            tuple[FlextInfraConfigModels.MakeVerbSpec, ...],
            m.Field(description="Repository-specific public Make verbs"),
        ] = ()
        script_dispatch: Annotated[
            FlextInfraConfigModels.ScriptDispatchSpec | None,
            m.Field(description="Optional script command dispatch contract"),
        ] = None

        makefile_custom_include: Annotated[
            t.NonEmptyStr,
            m.Field(description="Generated custom Make policy include directive"),
        ]
        orchestrated_verbs: Annotated[
            tuple[str, ...],
            m.Field(
                description="Repository-root gate verbs routed through orchestration"
            ),
        ] = ()
        workspace_cli_group: Annotated[
            t.NonEmptyStr,
            m.Field(description="CLI group used for workspace orchestration"),
        ]
        project_selection_conflict_error: Annotated[
            t.NonEmptyStr,
            m.Field(description="Mutually exclusive project selector error"),
        ]
        mypy_memory_limit_mb: Annotated[
            int, m.Field(gt=0, description="Generated Mypy address-space limit in MiB")
        ]
        mypy_timeout_seconds: Annotated[
            int, m.Field(gt=0, description="Generated Mypy wall-time limit in seconds")
        ]
        mypy_timeout_exit_code: Annotated[
            int, m.Field(gt=0, description="Wall-time limiter timeout exit code")
        ]
        mypy_signal_exit_offset: Annotated[
            int, m.Field(gt=0, description="Shell signal exit-code offset")
        ]
        prlimit_command: Annotated[
            t.NonEmptyStr, m.Field(description="Address-space limiter executable")
        ]
        prlimit_address_space_option: Annotated[
            t.NonEmptyStr, m.Field(description="Address-space limiter option")
        ]
        timeout_command: Annotated[
            t.NonEmptyStr, m.Field(description="Wall-time limiter executable")
        ]
        timeout_kill_after_seconds: Annotated[
            int, m.Field(gt=0, description="Forced-termination grace period")
        ]
        pytest_process_timeout_seconds: Annotated[
            int, m.Field(gt=0, description="Pytest process wall-time boundary")
        ]

    class BeadsConfigRenderSpec(_ConfigContract):
        """Field-only render input for the generated Beads ledger config."""

        issue_prefix: Annotated[
            t.NonEmptyStr,
            m.Field(description="Issue prefix from local config/beads.yaml"),
        ]
        endpoint_origin: Annotated[
            Literal["inherited_city"],
            m.Field(description="Gas City endpoint ownership projection"),
        ]
        endpoint_status: Annotated[
            Literal["verified"],
            m.Field(description="Gas City inherited endpoint status"),
        ]
        custom_issue_types: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(description="Union of project and required custom bead types"),
        ] = ()

    class BeadsMetadataRenderSpec(_ConfigContract):
        """Field-only render input for the generated Beads ledger marker.

        The marker carries portable storage and database identity plus the
        checkout's ledger identity. ``project_id`` is NOT invented here: it is
        read back from the checkout's own ``.beads/identity.toml`` so a
        regeneration preserves it. Omitting it made every ``make gen`` strip the
        key, and Beads then minted a fresh identity on next access — observed in
        rig ``gmn`` (commit 3e7ba1e), where the ledger identity changed from
        2b1a0582-… to e9a551fc-…. ``None`` means the checkout has no ledger
        identity yet, and Beads mints the first one.
        """

        database: Annotated[
            t.NonEmptyStr,
            m.Field(description="Dolt database from local config/beads.yaml"),
        ]
        project_id: Annotated[
            t.NonEmptyStr | None,
            m.Field(
                default=None,
                description=(
                    "Ledger identity read back from .beads/identity.toml; "
                    "None only before Beads has minted one"
                ),
            ),
        ] = None

    class GitignoreRenderSpec(_ConfigContract):
        """Typed, profile-filtered input for the generated Git ignore file."""

        gitignore_sections: Annotated[
            tuple[FlextInfraConfigModels.ScaffoldGitignoreSectionSpec, ...],
            m.Field(
                min_length=1,
                description="Canonical ignore sections applicable to one profile",
            ),
        ]

    class SgconfigRenderSpec(_ConfigContract):
        """Typed input for the generated ast-grep project config.

        Why (ai-hub-qwoc): a provider manifest can declare ``sgconfig.yml`` as a
        required surface, but no generator owned it, so the file was authored by
        hand in one repository and simply absent in another -- provider discovery
        then failed closed with ``missing declared file: sgconfig.yml``. The rule
        and fixture directories are declared here so every repository renders the
        same contract from the SSOT instead of a hand-written copy.
        """

        rule_dirs: Annotated[
            tuple[str, ...],
            m.Field(
                min_length=1,
                description="Directories holding this project's ast-grep rules",
            ),
        ]
        test_dirs: Annotated[
            tuple[str, ...],
            m.Field(
                default=(),
                description="Directories holding rule fixtures and snapshots",
            ),
        ]

    # Project creation metadata remains a typed manifest input.
    class ProjectSpec(_ConfigContract):
        """Deterministic project metadata required to materialize a new tree."""

        package_name: Annotated[
            t.NonEmptyStr, m.Field(description="Import package name")
        ]
        class_stem: Annotated[
            t.NonEmptyStr, m.Field(description="Canonical public facade class stem")
        ]
        namespace: Annotated[
            t.NonEmptyStr, m.Field(description="Nested c/t/p/m/u namespace")
        ]
        constant_name: Annotated[
            t.NonEmptyStr,
            m.Field(description="Configured project name exposed through constants"),
        ]
        namespace_attribute: Annotated[
            t.NonEmptyStr, m.Field(description="Private module namespace token")
        ]
        alias: Annotated[
            t.NonEmptyStr, m.Field(description="Canonical public instance alias")
        ]
        environment_prefix: Annotated[
            t.NonEmptyStr, m.Field(description="Project settings environment prefix")
        ]
        description: Annotated[
            t.NonEmptyStr, m.Field(description="Project description")
        ]
        license: Annotated[t.NonEmptyStr, m.Field(description="SPDX license id")]
        author_name: Annotated[
            t.NonEmptyStr, m.Field(description="Author display name")
        ]
        author_email: Annotated[t.NonEmptyStr, m.Field(description="Author email")]
        upstream: Annotated[
            t.NonEmptyStr, m.Field(description="Upstream FLEXT facade module")
        ]
        inherited_facets: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                default=(),
                description=(
                    "Upstream facets re-exported by the project root. The lazy-init "
                    "public-root planner reads this to decide which upstream "
                    "namespace names the generated root __init__ may re-export: a "
                    "facet is inherited when declared here or actually imported "
                    "from source. Without the field the planner cannot honour a "
                    "manifest declaration and re-exports only what source imports "
                    "prove -- which silently drops manifest-declared facets."
                ),
            ),
        ] = ()
        homepage: Annotated[t.NonEmptyStr, m.Field(description="Project homepage")]
        documentation: Annotated[
            t.NonEmptyStr, m.Field(description="Project documentation URL")
        ]
        repository_root_rel: Annotated[
            t.NonEmptyStr,
            m.Field(description="Declared relative path to the repository root"),
        ]
        year: Annotated[int, m.Field(ge=2025, description="Copyright year")]

    class MakeRenderContext(MakeCommandContext):
        """Typed input consumed by the generated Make surface."""

        make: Annotated[
            FlextInfraConfigModels.MakeSpec,
            m.Field(description="Generated Make command contract"),
        ]
        mypy_memory_limit_mb: Annotated[
            int, m.Field(gt=0, description="Generated Mypy address-space limit in MiB")
        ]
        mypy_timeout_seconds: Annotated[
            int, m.Field(gt=0, description="Generated Mypy wall-time limit in seconds")
        ]
        mypy_timeout_exit_code: Annotated[
            int, m.Field(gt=0, description="Wall-time limiter timeout exit code")
        ]
        mypy_signal_exit_offset: Annotated[
            int, m.Field(gt=0, description="Shell signal exit-code offset")
        ]
        prlimit_command: Annotated[
            t.NonEmptyStr, m.Field(description="Address-space limiter executable")
        ]
        prlimit_address_space_option: Annotated[
            t.NonEmptyStr, m.Field(description="Address-space limiter option")
        ]
        timeout_command: Annotated[
            t.NonEmptyStr, m.Field(description="Wall-time limiter executable")
        ]
        timeout_kill_after_seconds: Annotated[
            int, m.Field(gt=0, description="Forced-termination grace period")
        ]
        tooling_runtime: Annotated[
            FlextInfraModelsDepsToolSettings.ToolingRuntimeContext,
            m.Field(description="Resolved project/workspace tooling values"),
        ]

        dist: Annotated[t.NonEmptyStr, m.Field(description="Distribution name")]

        python_version: Annotated[
            t.NonEmptyStr, m.Field(description="Python major.minor tool value")
        ]
        uv_link_mode: Annotated[
            t.NonEmptyStr, m.Field(description="Configured uv installation link mode")
        ]
        uv_exclude_newer: Annotated[
            t.NonEmptyStr,
            m.Field(description="uv exclude-newer cooldown window for [tool.uv]"),
        ]
        dependency_cooldown_exclusions: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(description="Packages exempted from uv dependency cooldown"),
        ] = ()
        dependency_cooldown_overrides: Annotated[
            t.StrMapping,
            m.Field(
                default_factory=immutable_empty_mapping,
                description="Per-package cooldown cutoffs as RFC 3339 timestamps",
            ),
        ]
        ruff_per_file_ignores: Annotated[
            t.MappingKV[str, t.StrSequence],
            m.Field(
                default_factory=immutable_empty_mapping,
                description=(
                    "Effective Ruff exemptions: fleet policy composed with this "
                    "repository's own ManagedArtifacts overlay"
                ),
            ),
        ]
        make_profile: Annotated[
            FlextInfraConstantsCodegenProject.MakeProfile,
            m.Field(description="Generated Make execution profile"),
        ]
        repository_root_rel: Annotated[
            t.NonEmptyStr,
            m.Field(description="Relative path to the declared repository root"),
        ]
        makefile_custom_include: Annotated[
            str,
            m.Field(
                min_length=1,
                description=("Make directive that includes the custom Make surface"),
            ),
        ]
        declared_repositories: Annotated[
            tuple[str, ...],
            m.Field(description="Ordered workspace declared_repository paths"),
        ] = ()
        workspace_repositories: Annotated[
            tuple[FlextInfraConfigModels.RepositoryRef, ...],
            m.Field(description="Ordered workspace declared_repository records"),
        ] = ()
        workspace_gitlinks: Annotated[
            tuple[FlextInfraConfigModels.ManagedGitlinkSpec, ...],
            m.Field(description="Provider-resolved governed Git submodules"),
        ] = ()
        extra_verbs: Annotated[
            tuple[FlextInfraConfigModels.MakeVerbSpec, ...],
            m.Field(description="Repository-specific additional public Make verbs"),
        ] = ()
        script_dispatch: Annotated[
            FlextInfraConfigModels.ScriptDispatchSpec | None,
            m.Field(description="Opt-in script command-framework routing contract"),
        ] = None
        orchestrated_verbs: Annotated[
            tuple[str, ...],
            m.Field(
                description=(
                    "Gate verbs a workspace Makefile fans out across declared_repositories "
                    "through the generic workspace orchestrate primitive"
                )
            ),
        ] = ()
        workspace_cli_group: Annotated[
            str,
            m.Field(
                description=(
                    "CLI group name for the flext-infra workspace orchestrate route"
                )
            ),
        ] = ""
        project_selection_conflict_error: Annotated[
            t.NonEmptyStr,
            m.Field(description="Mutually exclusive project selector error"),
        ]

    class ProjectRenderContext(MakeRenderContext):
        """Complete typed input consumed by project scaffold templates."""

        @m.computed_field
        @property
        def repository_env_prefix(self) -> str:
            """Settings environment prefix derived from the distribution name."""
            return f"{self.dist.upper().replace('-', '_')}_"

        scaffold: Annotated[
            FlextInfraConfigModels.ScaffoldSpec,
            m.Field(description="New-project scaffold policy"),
        ]
        gitignore_sections: Annotated[
            tuple[FlextInfraConfigModels.ScaffoldGitignoreSectionSpec, ...],
            m.Field(
                min_length=1,
                description=(
                    "Canonical .gitignore sections derived from the artifact SSOT"
                ),
            ),
        ]
        dependency_profile: Annotated[
            FlextInfraConfigModels.ScaffoldDependencyProfileSpec,
            m.Field(description="Resolved upstream dependency profile"),
        ]
        tooling: Annotated[
            FlextInfraModelsDepsToolSettings.ToolConfigDocument,
            m.Field(description="Canonical validated tooling policy"),
        ]
        environment_path_prepends: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(description="Configured read-only PATH additions for direnv"),
        ] = ()
        beads_tool_selector: Annotated[
            t.NonEmptyStr, m.Field(description="Official Beads mise selector")
        ]
        beads_tool_version: Annotated[
            t.NonEmptyStr, m.Field(description="Exact Beads CLI version")
        ]
        beads_tool_prerelease: Annotated[
            bool,
            m.Field(description="Whether mise may resolve prerelease Beads versions"),
        ] = False
        beads_tool_minimum_release_age: Annotated[
            t.NonEmptyStr | None,
            m.Field(description="Per-tool Beads release quarantine override"),
        ] = None
        beads: Annotated[
            FlextInfraConfigModels.BeadsProjectSpec | None,
            m.Field(
                description=(
                    "Explicit repository-local Beads identity; only the ledger "
                    "templates read it, and only the observed route supplies it"
                )
            ),
        ] = None
        canonical_project_name: Annotated[
            t.NonEmptyStr, m.Field(description="Canonical PEP 621 project name")
        ]
        const_name: Annotated[
            t.NonEmptyStr, m.Field(description="Configured constant project name")
        ]
        package_name: Annotated[
            t.NonEmptyStr, m.Field(description="Python import package name")
        ]
        packaged_data_dirs: Annotated[
            t.StrSequence,
            m.Field(description="Generated root data directories shipped in wheels"),
        ]
        class_stem: Annotated[
            t.NonEmptyStr, m.Field(description="Public facade class stem")
        ]
        ns: Annotated[t.NonEmptyStr, m.Field(description="Public model namespace")]
        ns_attr: Annotated[
            t.NonEmptyStr, m.Field(description="Private namespace module token")
        ]
        alias: Annotated[t.NonEmptyStr, m.Field(description="Public instance alias")]
        env_prefix: Annotated[
            t.NonEmptyStr, m.Field(description="Settings environment prefix")
        ]
        upstream: Annotated[
            t.NonEmptyStr, m.Field(description="Upstream FLEXT facade module")
        ]
        inherited_facets: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                default=(),
                description=(
                    "Upstream facets re-exported by the project root; see the "
                    "RepositoryRef namesake for the lazy-init contract."
                ),
            ),
        ] = ()
        description: Annotated[
            t.NonEmptyStr, m.Field(description="Project description")
        ]
        version: Annotated[t.NonEmptyStr, m.Field(description="Project version")]
        license: Annotated[t.NonEmptyStr, m.Field(description="SPDX license id")]
        python_required_version: Annotated[
            t.NonEmptyStr, m.Field(description="PEP 440 project Python requirement")
        ]
        kubectl_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact kubectl toolchain version")
        ]
        helm_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact Helm toolchain version")
        ]
        kind_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact kind toolchain version")
        ]
        direnv_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Compatible direnv major.minor line")
        ]
        uv_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Compatible uv major.minor line")
        ]
        qlty_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact qlty code-quality version")
        ]
        node_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Compatible Node.js major.minor line")
        ]
        jscpd_version: Annotated[
            t.NonEmptyStr,
            _tool_version_field("Exact jscpd duplication detector version"),
        ]
        taplo_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact Taplo formatter version")
        ]
        ast_grep_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact ast-grep analyzer version")
        ]
        gitleaks_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact Gitleaks scanner version")
        ]
        scc_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact scc code-counter version")
        ]
        kubeconform_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Compatible kubeconform minor line")
        ]
        go_version: Annotated[
            t.NonEmptyStr, _tool_version_field("Exact Go runtime version")
        ]
        author_name: Annotated[
            t.NonEmptyStr, m.Field(description="Author display name")
        ]
        author_email: Annotated[t.NonEmptyStr, m.Field(description="Author email")]
        repository: Annotated[
            t.NonEmptyStr, m.Field(description="Project repository page URL")
        ]
        homepage: Annotated[t.NonEmptyStr, m.Field(description="Project homepage")]
        documentation: Annotated[
            t.NonEmptyStr, m.Field(description="Project documentation URL")
        ]
        flext_git_base_url: Annotated[
            t.NonEmptyStr, m.Field(description="FLEXT Git provider base URL")
        ]
        flext_git_branch: Annotated[
            t.NonEmptyStr, m.Field(description="FLEXT Git provider branch")
        ]
        repository_provider: Annotated[
            t.NonEmptyStr, m.Field(description="Repository provider catalog key")
        ]
        repository_git_url: Annotated[
            t.NonEmptyStr, m.Field(description="Canonical repository Git clone URL")
        ]
        repository_branch: Annotated[
            t.NonEmptyStr, m.Field(description="Canonical repository Git branch")
        ]
        year: Annotated[int, m.Field(description="Copyright year")]

    class WorkspaceBeadsServerSpec(_ConfigContract):
        """Optional Dolt connection declared by a versioned workspace manifest."""

        backend: Annotated[
            Literal["dolt"], m.Field(description="Workspace ledger storage engine")
        ]
        mode: Annotated[
            Literal["server"], m.Field(description="Workspace ledger connection mode")
        ]
        shared_server: Annotated[
            Literal[False],
            m.Field(description="Workspace ledger uses an external Dolt endpoint"),
        ] = False
        host: Annotated[t.NonEmptyStr, m.Field(description="Dolt server host")]
        port: Annotated[
            int, m.Field(ge=1, le=65535, description="Dolt server TCP port")
        ]
        user: Annotated[t.NonEmptyStr, m.Field(description="Dolt server user")]
        auto_commit: Annotated[
            Literal["off", "on", "batch"],
            m.Field(description="Dolt auto-commit policy"),
        ]

    class WorkspaceIntegrationSpec(_ConfigContract):
        """Workspace overlay for one provider integration branch."""

        provider: Annotated[
            t.NonEmptyStr, m.Field(description="Configured provider key")
        ]
        branch: Annotated[
            t.NonEmptyStr, m.Field(description="Workspace integration branch")
        ]
        organization: Annotated[
            t.NonEmptyStr | None,
            m.Field(description="Optional provider organization override"),
        ] = None
        base_url: Annotated[
            t.NonEmptyStr | None,
            m.Field(description="Optional provider base URL override"),
        ] = None

    class RepositoryPolicyOverlaySpec(_ConfigContract):
        """Bounded per-project policy declared by a workspace manifest."""

        project: Annotated[
            t.NonEmptyStr, m.Field(description="Canonical project distribution")
        ]
        beads_enabled: Annotated[
            bool, m.Field(description="Whether the repository participates in Beads")
        ] = False
        ci_enabled: Annotated[
            bool, m.Field(description="Whether conform owns the CI surface")
        ] = True
        ci_matrix_auto_run: Annotated[
            bool, m.Field(description="Whether the CI matrix runs automatically")
        ] = False
        extra_ignored_patterns: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(description="Repository-local generated ignore patterns"),
        ] = ()

    class WorkspaceExclusionSpec(_ConfigContract):
        """One explicitly excluded workspace-relative path."""

        path: Annotated[Path, m.Field(description="Workspace-relative path")]
        reason: Annotated[t.NonEmptyStr, m.Field(description="Exclusion rationale")]

    class WorkspaceManifestSpec(_ConfigContract):
        """Complete versioned input contract for ``config/workspace.yaml``."""

        version: Annotated[
            int,
            m.Field(
                ge=FlextInfraConstantsCodegenProject.WORKSPACE_MANIFEST_VERSION,
                le=FlextInfraConstantsCodegenProject.WORKSPACE_MANIFEST_VERSION,
                description="Workspace manifest schema version",
            ),
        ]
        name: Annotated[t.NonEmptyStr, m.Field(description="Workspace name")]
        ledger_id: Annotated[
            t.NonEmptyStr | None,
            m.Field(description="Optional workspace ledger database identity"),
        ] = None
        ledger_prefix: Annotated[
            t.NonEmptyStr | None,
            m.Field(description="Optional workspace issue-prefix identity"),
        ] = None
        beads_server: Annotated[
            FlextInfraConfigModels.WorkspaceBeadsServerSpec | None,
            m.Field(description="Optional workspace-local Dolt connection"),
        ] = None
        repository: Annotated[
            FlextInfraConfigModels.RepositoryRef,
            m.Field(description="Root repository declaration"),
        ]
        project: Annotated[
            FlextInfraConfigModels.ProjectSpec | None,
            m.Field(description="Optional project creation metadata"),
        ] = None
        members: Annotated[
            tuple[FlextInfraConfigModels.RepositoryRef, ...],
            m.Field(description="Declared member repository contracts"),
        ] = ()
        external_dependency_paths: Annotated[
            tuple[Path, ...], m.Field(description="Declared external dependency paths")
        ] = ()
        content_only: Annotated[
            tuple[Path, ...], m.Field(description="Content-only Gitlink paths")
        ] = ()
        exclusions: Annotated[
            tuple[FlextInfraConfigModels.WorkspaceExclusionSpec, ...],
            m.Field(description="Explicit workspace exclusions"),
        ] = ()
        integration: Annotated[
            FlextInfraConfigModels.WorkspaceIntegrationSpec | None,
            m.Field(description="Optional integration provider overlay"),
        ] = None
        repository_policy_overlays: Annotated[
            tuple[FlextInfraConfigModels.RepositoryPolicyOverlaySpec, ...],
            m.Field(description="Repository-local policy overlays"),
        ] = ()

        @u.model_validator(mode="after")
        def _validate_references(self) -> Self:
            """Reject ambiguous paths and policy references in the full document."""
            external_paths = (*self.external_dependency_paths, *self.content_only)
            invalid_paths = tuple(
                path
                for path in external_paths
                if path.is_absolute() or not path.parts or ".." in path.parts
            )
            if invalid_paths:
                msg = "workspace dependency paths must be relative: " + ", ".join(
                    path.as_posix() for path in invalid_paths
                )
                raise ValueError(msg)
            member_paths = tuple(item.path for item in self.members)
            if len(set(member_paths)) != len(member_paths):
                msg = "workspace member paths must be unique"
                raise ValueError(msg)
            if set(member_paths).intersection(external_paths):
                msg = "workspace members cannot also be external dependencies"
                raise ValueError(msg)
            projects = tuple(item.project for item in self.repository_policy_overlays)
            if len(set(projects)) != len(projects):
                msg = "repository policy overlays must be unique"
                raise ValueError(msg)
            repository_names = {
                item.distribution for item in (self.repository, *self.members)
            }
            unknown_projects = set(projects).difference(repository_names)
            if unknown_projects:
                msg = (
                    "repository policy overlays reference unknown projects: "
                    + ", ".join(sorted(unknown_projects))
                )
                raise ValueError(msg)
            return self

    class WorkspaceSpec(_ConfigContract):
        """Local identity plus topology read from this repository's Git inputs."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Workspace name")]
        beads: Annotated[
            FlextInfraConfigModels.BeadsProjectSpec | None,
            m.Field(
                description=(
                    "Repository-local Beads identity. The observed loader always "
                    "resolves it; the declaration-only projection loader leaves it "
                    "absent because no declaration carries a ledger identity."
                )
            ),
        ] = None
        repository: Annotated[
            FlextInfraConfigModels.RepositoryRef,
            m.Field(description="Local repository Git contract"),
        ]
        project: Annotated[
            FlextInfraConfigModels.ProjectSpec | None,
            m.Field(description="Metadata required only when materializing a new tree"),
        ] = None
        declared_repositories: Annotated[
            tuple[FlextInfraConfigModels.RepositoryRef, ...],
            m.Field(description="Direct governed repositories from local .gitmodules"),
        ] = ()
        external_dependency_paths: Annotated[
            tuple[Path, ...],
            m.Field(description="Observed external or fork Git submodule paths"),
        ] = ()
        repository_policy_overlays: Annotated[
            tuple[FlextInfraConfigModels.RepositoryPolicyOverlaySpec, ...],
            m.Field(
                description=(
                    "Repository-local policy overlays carried from the declared "
                    "manifest so scoped projection keeps per-repository policy"
                )
            ),
        ] = ()

        @u.model_validator(mode="after")
        def _validate_topology_paths(self) -> Self:
            """Reject duplicate, ambiguous, or escaping topology paths."""
            invalid_external_paths = tuple(
                path
                for path in self.external_dependency_paths
                if path.is_absolute() or not path.parts or ".." in path.parts
            )
            if invalid_external_paths:
                msg = (
                    "external dependency paths must be workspace-relative: "
                    f"{', '.join(path.as_posix() for path in invalid_external_paths)}"
                )
                raise ValueError(msg)
            if len(set(self.external_dependency_paths)) != len(
                self.external_dependency_paths
            ):
                msg = "external dependency paths must be unique"
                raise ValueError(msg)
            declared_paths = {item.path for item in self.declared_repositories}
            if len(declared_paths) != len(self.declared_repositories):
                msg = "declared_repository paths must be unique"
                raise ValueError(msg)
            overlap = declared_paths.intersection(self.external_dependency_paths)
            if overlap:
                msg = (
                    "external dependencies cannot also be governed declared_repositories: "
                    f"{', '.join(sorted(path.as_posix() for path in overlap))}"
                )
                raise ValueError(msg)
            return self

    # The artifact list is the SINGLE SSOT for
    # ephemeral/generated resources; VS Code excludes and source_scan ignores
    # are derived projections, never re-declared in YAML.
    class CodegenArtifactSpec(_ConfigContract):
        """One ephemeral/generated resource every ignore/exclude derives from."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Basename of the resource")]
        is_dir: Annotated[bool, m.Field(description="Directory (vs file) resource")] = (
            True
        )
        vscode_exclude: Annotated[
            bool, m.Field(description="Feed VS Code files.exclude + search.exclude")
        ] = True
        watch_exclude: Annotated[
            bool, m.Field(description="Feed VS Code files.watcherExclude")
        ] = True
        gitignore: Annotated[
            bool, m.Field(description="Feed the Python/tool section of .gitignore")
        ] = True
        source_scan_ignore: Annotated[
            bool, m.Field(description="Feed source_scan.ignored_resources")
        ] = False

    class CodegenVscodeSpec(_ConfigContract):
        """Fully modeled content of the ``vscode`` section of ``config/codegen.yaml``."""

        scalar_settings: Annotated[
            Mapping[str, str | bool],
            m.Field(description="VS Code scalar keys enforced on every project"),
        ]
        list_settings: Annotated[
            Mapping[str, tuple[str, ...]],
            m.Field(description="VS Code list keys enforced on every project"),
        ]
        map_union_settings: Annotated[
            Mapping[str, Mapping[str, str | bool]],
            m.Field(description="VS Code map keys union-merged over project settings"),
        ]

    class CodegenConfigSpec(_ConfigContract):
        """Fully modeled content of ``config/codegen.yaml``."""

        version: Annotated[int, m.Field(ge=1, description="Config schema version")]
        toolchain: Annotated[
            FlextInfraConfigModels.ToolchainSpec,
            m.Field(description="Exact generated toolchain"),
        ]
        github_actions: Annotated[
            Mapping[str, FlextInfraConfigModels.GithubActionPinSpec],
            m.Field(description="Immutable GitHub Action catalog"),
        ]
        checkout_submodules: Annotated[
            t.NonEmptyStr,
            m.Field(
                default="false",
                pattern=r"^(true|false|recursive)$",
                description=(
                    "Default actions/checkout submodules mode for every "
                    "generated workflow. 'false' keeps CI green on projects "
                    "whose submodules are private: the default GITHUB_TOKEN "
                    "cannot clone sibling private repositories and "
                    "'recursive' aborts the job at checkout"
                ),
            ),
        ]
        checkout_submodules_overrides: Annotated[
            Mapping[str, str],
            m.Field(
                default_factory=immutable_empty_mapping,
                description=(
                    "Per-distribution override of checkout_submodules, for "
                    "projects that really do exercise their declared_repositories in CI"
                ),
            ),
        ]
        ci_private_submodules: Annotated[
            Mapping[str, FlextInfraConfigModels.CiPrivateSubmodulesSpec],
            m.Field(
                default_factory=immutable_empty_mapping,
                description=(
                    "Per-distribution private submodule deploy-key contracts "
                    "rendered into generated CI before make setup"
                ),
            ),
        ]
        ci_system_packages: Annotated[
            Mapping[str, tuple[t.NonEmptyStr, ...]],
            m.Field(
                default_factory=immutable_empty_mapping,
                description=(
                    "Per-distribution runner packages (Ubuntu apt names) the "
                    "generated CI installs before the gates run"
                ),
            ),
        ]
        sgconfig: Annotated[
            FlextInfraConfigModels.SgconfigRenderSpec,
            m.Field(description="Canonical ast-grep project contract for every repo"),
        ]
        uv_exclude_dependencies: Annotated[
            tuple[FlextInfraConfigModels.UvScopedDependencyExclusionSpec, ...],
            m.Field(description="Project-scoped official uv dependency exclusions"),
        ] = ()
        infra_repository: Annotated[
            FlextInfraConfigModels.RepositorySourceSpec,
            m.Field(description="Canonical infrastructure repository identity"),
        ]
        providers: Annotated[
            tuple[FlextInfraConfigModels.ProviderSpec, ...],
            m.Field(min_length=1, description="Ordered FLEXT-owned Git providers"),
        ]
        branch_policy: Annotated[
            FlextInfraConfigModels.BranchPolicySpec,
            m.Field(description="Global governed branch ancestry policy"),
        ]
        profiles: Annotated[
            tuple[FlextInfraConfigModels.ProfileSpec, ...],
            m.Field(description="Ordered Make profiles"),
        ]
        make: Annotated[
            FlextInfraConfigModels.MakeSpec,
            m.Field(description="Canonical Make contract"),
        ]
        vscode: Annotated[
            FlextInfraConfigModels.CodegenVscodeSpec,
            m.Field(description="Canonical VS Code settings merge contract"),
        ]
        artifacts: Annotated[
            tuple[FlextInfraConfigModels.CodegenArtifactSpec, ...],
            m.Field(
                min_length=1,
                description=(
                    "Ephemeral/generated artifact SSOT; every ignore/exclude "
                    "projection derives from this list"
                ),
            ),
        ]
        layout: Annotated[
            FlextInfraModelsLayout.LayoutSpec,
            m.Field(
                description=(
                    "Declarative project-layout conformance contract consumed "
                    "by the layout engine and the layout quality gate"
                )
            ),
        ]

        @u.model_validator(mode="after")
        def _validate_infrastructure_provider(self) -> Self:
            """Require the tool distribution owner to resolve exactly once.

            ``infra_repository`` is the single owner of that identity. The
            separate ``infrastructure_provider`` key that duplicated it is
            retired (flext-infra#576), so the check reads the surviving owner.
            """
            matches = tuple(
                provider
                for provider in self.providers
                if provider.name == self.infra_repository.provider
            )
            if len(matches) != 1:
                msg = (
                    "infra_repository.provider must resolve exactly once in "
                    f"providers: {self.infra_repository.provider}"
                )
                raise ValueError(msg)
            return self

        @m.computed_field
        @property
        def vscode_files_exclude_map(self) -> Mapping[str, bool]:
            """Derived VS Code ``files.exclude`` entries from the artifact SSOT."""
            return {
                f"**/{artifact.name}": True
                for artifact in self.artifacts
                if artifact.vscode_exclude
            }

        @m.computed_field
        @property
        def vscode_watcher_exclude_map(self) -> Mapping[str, bool]:
            """Derived VS Code ``files.watcherExclude`` entries from the SSOT."""
            return {
                f"**/{artifact.name}/**": True
                for artifact in self.artifacts
                if artifact.watch_exclude
            }

        @m.computed_field
        @property
        def vscode_search_exclude_map(self) -> Mapping[str, bool]:
            """Derived VS Code ``search.exclude`` entries from the artifact SSOT."""
            return dict(self.vscode_files_exclude_map)

        @m.computed_field
        @property
        def source_scan_ignored(self) -> tuple[str, ...]:
            """Derived ``source_scan.ignored_resources`` names from the SSOT."""
            return tuple(
                artifact.name
                for artifact in self.artifacts
                if artifact.source_scan_ignore
            )

        # The canonical .gitignore body is ONE computed
        # projection — the artifact SSOT feeds the Python/build section and the
        # static scaffold sections carry only what the SSOT cannot express
        # (file globs, secrets, editor/OS noise). Per-project exception fields
        # (extra_ignored / allowed dirs) land in their typed owner;
        # this projection is the seam they will extend.
        @m.computed_field
        @property
        def gitignore_sections(
            self,
        ) -> tuple[FlextInfraConfigModels.ScaffoldGitignoreSectionSpec, ...]:
            """Derived canonical ``.gitignore`` sections (SSOT order, deduplicated).

            Ignore files are order-sensitive: a pattern placed before a
            catch-all such as ``/*`` is dead, and a directory ignored before
            its own ``!`` negation is never re-allowed. The declared sections
            are therefore emitted in their declared order, and derived artifact
            patterns are appended -- never prepended -- so a whitelist policy
            expressed in the SSOT survives the projection intact.
            """
            scaffold_sections = self.scaffold.gitignore_sections
            # A declared section may already govern a derived artifact, in
            # either direction: a whitelist re-allows `.agents/` with `!`, so
            # appending a bare `.agents/` ignore would contradict the declared
            # policy. Only artifacts the SSOT never mentions are appended.
            governed = {
                pattern.lstrip("!")
                for section in scaffold_sections
                for pattern in section.patterns
            }
            managed_allowed: t.MutableSequenceOf[str] = []
            declared_patterns = {
                pattern for section in scaffold_sections for pattern in section.patterns
            }
            for managed in self.managed_files:
                if (
                    managed.policy
                    == FlextInfraConstantsSharedInfra.MANAGED_FILE_POLICY_DELEGATED
                ):
                    continue
                parts = managed.path.parts
                candidates = [
                    *(f"!{'/'.join(parts[:depth])}/" for depth in range(1, len(parts))),
                    f"!{managed.path.as_posix()}",
                ]
                managed_allowed.extend(
                    candidate
                    for candidate in candidates
                    if candidate not in declared_patterns
                    and candidate not in managed_allowed
                )
            derived: t.MutableSequenceOf[str] = []
            for pattern in self.gitignore_artifact_patterns:
                if pattern not in governed and pattern not in derived:
                    derived.append(pattern)
            sections: t.MutableSequenceOf[
                FlextInfraConfigModels.ScaffoldGitignoreSectionSpec
            ] = []
            # Declared sections are emitted verbatim. Cross-section dedup is
            # unsound for ignore files: repeating `.beads/*` after an
            # intervening `!.beads/` is what keeps the directory scanned, so
            # dropping the repeat silently un-ignores its contents.
            sections.extend(scaffold_sections)
            # Derived artifacts are appended as their own trailing section: an
            # ignore file is evaluated in order, so injecting them into the
            # first section would place them before any `!` negation the policy
            # declares later and silently un-ignore governed paths.
            if derived:
                sections.append(
                    FlextInfraConfigModels.ScaffoldGitignoreSectionSpec(
                        name=FlextInfraConstantsSharedInfra.GITIGNORE_DERIVED_SECTION_NAME,
                        patterns=tuple(derived),
                    )
                )
            if managed_allowed:
                sections.append(
                    FlextInfraConfigModels.ScaffoldGitignoreSectionSpec(
                        name=FlextInfraConstantsSharedInfra.GITIGNORE_MANAGED_SECTION_NAME,
                        patterns=tuple(managed_allowed),
                    )
                )
            return tuple(sections)

        @m.computed_field
        @property
        def gitignore_artifact_patterns(self) -> tuple[str, ...]:
            """Derived ``.gitignore`` artifact patterns from the SSOT (stable order)."""
            return tuple(
                f"{artifact.name}/" if artifact.is_dir else artifact.name
                for artifact in self.artifacts
                if artifact.gitignore
            )

        managed_files: Annotated[
            tuple[FlextInfraConfigModels.ManagedFileSpec, ...],
            m.Field(description="Files owned by conform"),
        ]
        scaffold: Annotated[
            FlextInfraConfigModels.ScaffoldSpec,
            m.Field(description="Typed new-project scaffold policy"),
        ]
        templates: Annotated[
            FlextInfraConfigModels.TemplatesSpec,
            m.Field(description="New-project-only scaffold template manifest"),
        ]
        # Operator law: flext-infra owns generic conform policy only. The set
        # of projects it serves is NOT its knowledge — each repository's own
        # .gitmodules is the read-only topology authority.

        @u.model_validator(mode="after")
        def _validate_github_artifact_ownership(self) -> Self:
            """Require one full-managed conform owner for every GitHub template."""
            github_templates = tuple(
                Path(entry.destination)
                for entry in self.templates.entries
                if Path(entry.destination).parts[:1] == (".github",)
            )
            github_managed = tuple(
                managed
                for managed in self.managed_files
                if managed.path.parts[:1] == (".github",)
            )
            template_paths = set(github_templates)
            managed_paths = {managed.path for managed in github_managed}
            duplicate_templates = len(github_templates) != len(template_paths)
            duplicate_managed = len(github_managed) != len(managed_paths)
            if duplicate_templates or duplicate_managed:
                msg = (
                    "GitHub artifacts must have exactly one template and managed owner"
                )
                raise ValueError(msg)
            if template_paths != managed_paths:
                missing_owners = sorted(
                    path.as_posix() for path in template_paths - managed_paths
                )
                missing_templates = sorted(
                    path.as_posix() for path in managed_paths - template_paths
                )
                msg = (
                    "GitHub template/managed ownership mismatch: "
                    f"missing owners={missing_owners}, "
                    f"missing templates={missing_templates}"
                )
                raise ValueError(msg)
            non_full = sorted(
                managed.path.as_posix()
                for managed in github_managed
                if managed.policy
                != FlextInfraConstantsSharedInfra.MANAGED_FILE_POLICY_FULL
            )
            if non_full:
                msg = f"GitHub artifacts must be full-managed: {non_full}"
                raise ValueError(msg)
            return self

    # Production source
    # selection is modeled once so iteration, Rope, and census share one SSOT.
    class SourceScanSpec(_ConfigContract):
        """Canonical production roots and recursively ignored directories."""

        roots: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(description="Ordered production source directory names"),
        ]

    class StaticRule(_ConfigContract):
        """Shared immutable metadata for one Rope static-analysis rule."""

        kind: t.NonEmptyStr = m.Field(description="Violation kind")
        detail: t.NonEmptyStr = m.Field(description="Root-cause explanation")

    class StaticImportModuleRule(StaticRule):
        """Reject one imported module outside its configured owning project."""

        operator: Literal["import_module"] = m.Field(description="Operator")
        module: t.NonEmptyStr = m.Field(description="Rejected module root")
        owner_project: t.NonEmptyStr | None = m.Field(
            default=None, description="Permitted owning project"
        )

    class StaticImportMemberRule(StaticRule):
        """Reject one member imported from a configured module."""

        operator: Literal["import_member"] = m.Field(description="Operator")
        module: t.NonEmptyStr = m.Field(description="Import source module")
        member: t.NonEmptyStr = m.Field(description="Rejected imported member")

    class StaticAttributeRule(StaticRule):
        """Reject one member accessed through a semantically imported module alias."""

        operator: Literal["attribute"] = m.Field(description="Operator")
        module: t.NonEmptyStr = m.Field(description="Imported module")
        member: t.NonEmptyStr = m.Field(description="Rejected attribute")

    class StaticCallRule(StaticRule):
        """Reject calls to one bare callable name."""

        operator: Literal["call"] = m.Field(description="Operator")
        name: t.NonEmptyStr = m.Field(description="Rejected callable")

    class StaticCallKeywordRule(StaticRule):
        """Require one keyword in calls to a configured callable."""

        operator: Literal["call_keyword"] = m.Field(description="Operator")
        name: t.NonEmptyStr = m.Field(description="Callable")
        keyword: t.NonEmptyStr = m.Field(description="Required keyword")

    class StaticAnnotationRule(StaticRule):
        """Reject one identifier anywhere in an annotation."""

        operator: Literal["annotation"] = m.Field(description="Operator")
        name: t.NonEmptyStr = m.Field(description="Rejected annotation identifier")

    class StaticBareExceptRule(StaticRule):
        """Reject an exception handler without an exception contract."""

        operator: Literal["bare_except"] = m.Field(description="Operator")

    class StaticAnnotatedStringRule(StaticRule):
        """Reject an annotated target assigned directly to a string literal."""

        operator: Literal["annotated_string"] = m.Field(description="Operator")
        name: t.NonEmptyStr = m.Field(description="Rejected assignment target")

    class StaticCommentRule(StaticRule):
        """Reject one marker only when Rope classifies its region as a comment."""

        operator: Literal["comment"] = m.Field(description="Operator")
        marker: t.NonEmptyStr = m.Field(description="Rejected comment marker")

    type StaticRuleSpec = Annotated[
        StaticImportModuleRule
        | StaticImportMemberRule
        | StaticAttributeRule
        | StaticCallRule
        | StaticCallKeywordRule
        | StaticAnnotationRule
        | StaticBareExceptRule
        | StaticAnnotatedStringRule
        | StaticCommentRule,
        m.Field(discriminator="operator"),
    ]

    class StaticEnforcementSpec(_ConfigContract):
        """Complete validated static policy evaluated only through Rope facts."""

        rules: Annotated[
            tuple[FlextInfraConfigModels.StaticRuleSpec, ...],
            m.Field(min_length=1, description="Ordered static-analysis rules"),
        ]

    class ReleasePolicySpec(_ConfigContract):
        """The release protocol's declared data: who publishes, what bumps, where.

        Why (aihub-ioijy.9): `ReleaseOrchestrator._build_targets` hardcoded
        `project.name.startswith("flext-")`, so any consumer of this release
        engine whose distribution is not named `flext-*` resolved zero targets
        and died with "release build selected no publishable projects".
        Publishable membership is project policy, not a naming convention.

        `bump_types` maps a Conventional Commits type found in a merged
        pull-request title to the bump it earns; a type absent from the map
        releases nothing, and `!` in the title always earns a major bump. The
        Conventional Commits defaults are the typed default, so a consumer
        repository declares only what differs.
        """

        # The bump map is consumed as enum members by the strict release plan,
        # so the contract base's value coercion is switched off here.
        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            strict=False, frozen=True, extra="forbid", use_enum_values=False
        )

        publishable_prefixes: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                default=(),
                description=(
                    "Distribution-name prefixes eligible for build/publish. "
                    "Empty means every resolved project is eligible."
                ),
            ),
        ]
        bump_types: Annotated[
            Mapping[t.NonEmptyStr, FlextInfraConstantsRelease.VersionBump],
            m.Field(
                default_factory=lambda: {
                    "feat": FlextInfraConstantsRelease.VersionBump.MINOR,
                    "fix": FlextInfraConstantsRelease.VersionBump.PATCH,
                    "perf": FlextInfraConstantsRelease.VersionBump.PATCH,
                },
                description="Conventional Commits type -> semantic version bump",
            ),
        ]
        publish_url: Annotated[
            t.NonEmptyStr,
            m.Field(
                default="https://upload.pypi.org/legacy/",
                description="Package index upload endpoint for verified artifacts",
            ),
        ]
        build_constraints: Annotated[
            tuple[FlextInfraConfigModels.BuildConstraintSpec, ...],
            m.Field(
                min_length=1,
                description=(
                    "Hash-pinned build-backend requirements every release artifact "
                    "is built with; projected to config/build-constraints.txt"
                ),
            ),
        ]

    class BuildConstraintSpec(_ConfigContract):
        """One hash-pinned build requirement (``uv build --require-hashes``)."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Distribution name")]
        version: Annotated[t.NonEmptyStr, m.Field(description="Exact version")]
        hashes: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(min_length=1, description="Accepted sha256 digests"),
        ]

    class ReleasePolicyRenderSpec(_ConfigContract):
        """Typed input consumed by the generated release policy files."""

        build_constraints: Annotated[
            tuple[FlextInfraConfigModels.BuildConstraintSpec, ...],
            m.Field(min_length=1, description="Pins rendered into the constraints"),
        ]

    # This
    # field-only namespace is the sole validated owner exposed as config.Infra.
    class Infra(_ConfigContract):
        """Complete flext-infra configuration namespace."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Project distribution name")]
        initial_project_version: Annotated[
            t.NonEmptyStr,
            m.Field(
                description=(
                    "Version seeded into a newly scaffolded project's pyproject; "
                    "from then on the release protocol is the only writer"
                )
            ),
        ]
        codegen: Annotated[
            FlextInfraConfigModels.CodegenConfigSpec,
            m.Field(description="Unified project and workspace codegen contract"),
        ]
        tooling: Annotated[
            FlextInfraModelsDepsToolSettings.ToolConfigDocument,
            m.Field(description="Validated lint, typecheck, and scaffold policy"),
        ]
        source_scan: Annotated[
            FlextInfraConfigModels.SourceScanSpec,
            m.Field(description="Production-only source discovery contract"),
        ]
        release: Annotated[
            FlextInfraConfigModels.ReleasePolicySpec,
            m.Field(description="Release protocol policy: eligibility, bumps, index"),
        ]
        # Static policy is validated data, never detector code.
        enforcement: Annotated[
            FlextInfraConfigModels.StaticEnforcementSpec,
            m.Field(description="Rope-only static enforcement policy"),
        ]

    class Root(_ConfigContract):
        """Root payload deep-merged from flext-infra config files."""

        Infra: Annotated[
            FlextInfraConfigModels.Infra,
            m.Field(description="Validated flext-infra namespace"),
        ]

    class UvEnvironmentPlan(_ConfigContract):
        """One deterministic uv environment operation plan."""

        project_root: Annotated[Path, m.Field(description="Selected project root")]
        environment_root: Annotated[
            Path, m.Field(description="Project supplying the active .venv")
        ]
        lock_path: Annotated[Path, m.Field(description="Required versioned uv.lock")]
        python_version: Annotated[
            t.NonEmptyStr, m.Field(description="Mise/Python version selector")
        ]
        groups: Annotated[
            tuple[str, ...],
            m.Field(description="Ordered dependency groups synchronized by setup"),
        ]
        editable_repositories: Annotated[
            tuple[FlextInfraConfigModels.RepositoryRef, ...],
            m.Field(description="Local repositories overlaid after locked sync"),
        ] = ()

    class BranchAncestryRef(_ConfigContract):
        """One exact branch or registered worktree ancestry observation."""

        reference: Annotated[
            t.NonEmptyStr, m.Field(description="Git ref or worktree identity")
        ]
        sha: Annotated[
            t.NonEmptyStr, m.Field(description="Observed commit object identifier")
        ]
        excluded: Annotated[
            bool, m.Field(description="Whether typed technical policy excludes the ref")
        ]
        ancestor: Annotated[
            bool | None,
            m.Field(description="Baseline ancestry verdict; None when excluded"),
        ]

    class BranchAncestryPlan(_ConfigContract):
        """Bounded ancestry inventory for one governed repository."""

        repository_root: Annotated[
            Path, m.Field(description="Governed repository root")
        ]
        baseline_reference: Annotated[
            t.NonEmptyStr, m.Field(description="Provider-owned remote baseline ref")
        ]
        baseline_sha: Annotated[
            t.NonEmptyStr, m.Field(description="Resolved baseline commit")
        ]
        references: Annotated[
            tuple[FlextInfraConfigModels.BranchAncestryRef, ...],
            m.Field(description="Local, remote, and worktree ancestry inventory"),
        ]

    class WorkspaceEnvironmentCliRequest(_ConfigContract):
        """CLI-safe request for one Python workspace environment sync."""

        repository_root: Annotated[
            Path, m.Field(description="Repository root receiving the sync")
        ]
        apply: Annotated[
            bool, m.Field(description="Write changes instead of reporting them")
        ] = True
        force: Annotated[
            bool, m.Field(description="Replace custom files with generated content")
        ] = False
        allow_direnv: Annotated[
            bool,
            m.Field(description="Authorize the rendered .envrc after an applied sync"),
        ] = True

    class WorkspaceEnvironmentSyncRequest(_ConfigContract):
        """Validated internal request for one workspace environment sync."""

        repository_root: Annotated[
            Path, m.Field(description="Workspace root receiving the sync")
        ]
        apply: Annotated[
            bool, m.Field(description="Write changes instead of reporting them")
        ] = True
        force: Annotated[
            bool, m.Field(description="Replace custom files with generated content")
        ] = False
        beads: Annotated[
            FlextInfraConfigModels.BeadsWorkspaceEnvironmentSpec | None,
            m.Field(
                exclude=True,
                description=(
                    "Programmatic-only selection of generated beads-workspace "
                    ".envrc instead of the Python package environment"
                ),
            ),
        ] = None
        allow_direnv: Annotated[
            bool,
            m.Field(
                description=(
                    "Run `direnv allow` for the workspace after a successful "
                    "applied sync so managed roots never carry a stale allow"
                )
            ),
        ] = True

    class BeadsWorkspaceEnvironmentSpec(_ConfigContract):
        """Declarative contract for one generated beads-workspace .envrc.

        Defaults encode the canonical Gas City + Beads wiring: the sync owns
        the file end to end while every credential-free fact stays declarative
        here, and the single identity variable (``AGENTS_GAS_CITY_ROOT``) keeps
        failing loudly when the canonical checkout is not declared.
        """

        environment_sources: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(description="Environment files sourced on activation"),
        ] = ("$HOME/.config/environment.d/projects/agent-tools.envrc",)
        identity_var: Annotated[
            t.NonEmptyStr,
            m.Field(description="Required variable naming the Gas City checkout"),
        ] = "AGENTS_GAS_CITY_ROOT"
        city_state_relpath: Annotated[
            t.NonEmptyStr,
            m.Field(description="Gas City Dolt state path relative to the checkout"),
        ] = ".gc/runtime/packs/dolt/dolt-state.json"
        beads_metadata_relpath: Annotated[
            t.NonEmptyStr,
            m.Field(description="Beads metadata path relative to the workspace"),
        ] = ".beads/metadata.json"
        unset_vars: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(description="Inherited orchestration variables cleared on entry"),
        ] = (
            "GT_ROOT",
            "GT_TOWN_ROOT",
            "BEADS_DIR",
            "BEADS_DOLT_PORT",
            "BEADS_DOLT_DATA_DIR",
            "BEADS_DOLT_SHARED_SERVER",
        )

    class WorkspaceEnvironmentSyncResult(_ConfigContract):
        """Outcome of one workspace environment sync."""

        changed_files: Annotated[
            tuple[Path, ...],
            m.Field(description="Environment files created, updated, or removed"),
        ] = ()

        @m.computed_field
        @property
        def changed(self) -> bool:
            """Whether the sync altered any environment file."""
            return bool(self.changed_files)

    class CodegenConformSurfaceContract(m.Value):
        """Typed ownership contract for one requested conformance surface."""

        # Why: leaf conform planning contract lives on m.Infra only (not nested in services).
        destinations: Annotated[
            frozenset[str] | None,
            m.Field(description="Output paths selected for conformance planning"),
        ] = None
        complete_governed: Annotated[
            bool, m.Field(description="Whether every governed output is represented")
        ] = False
        dependencies_only: Annotated[
            bool, m.Field(description="Whether planning is dependency-only")
        ] = False
        delegates: Annotated[
            bool, m.Field(description="Whether delegated templates are planned")
        ] = True
        pyproject: Annotated[
            bool, m.Field(description="Whether project metadata is planned")
        ] = True
        templates: Annotated[
            bool, m.Field(description="Whether managed templates are planned")
        ] = True
        custom: Annotated[
            bool, m.Field(description="Whether custom Make policy is planned")
        ] = True

    class CodegenConformRequest(_ConfigContract):
        """Validated public request for ``flext-infra codegen conform``."""

        root: Annotated[Path, m.Field(description="Repository or repository root")]
        what: Annotated[
            FlextInfraConstantsCodegenProject.CodegenConformSurface,
            m.Field(description="Managed file selection"),
        ] = FlextInfraConstantsCodegenProject.CodegenConformSurface.ALL
        scope: Annotated[
            FlextInfraConstantsCodegenProject.CodegenConformScope,
            m.Field(description="Repository selection scope"),
        ] = FlextInfraConstantsCodegenProject.CodegenConformScope.SELF
        mode: Annotated[
            FlextInfraConstantsCodegenProject.CodegenConformMode,
            m.Field(description="Read-only check or atomic apply"),
        ] = FlextInfraConstantsCodegenProject.CodegenConformMode.CHECK

    class CodegenArtifactComposition(_ConfigContract):
        """Rendered artifact plus the exact source states used to compose it."""

        rendered: Annotated[
            str, m.Field(description="Fully composed managed-file content")
        ]
        source_states: Annotated[
            tuple[m.Cli.AtomicFileState, ...],
            m.Field(description="Ordered immutable sources consumed by composition"),
        ] = ()

    class CodegenFilePlan(_ConfigContract):
        """Expected content and current state for one managed file."""

        path: Annotated[Path, m.Field(description="Absolute managed file path")]
        rendered: Annotated[str, m.Field(description="Fully rendered expected content")]
        expected_sha256: Annotated[
            t.NonEmptyStr, m.Field(description="SHA-256 of expected content")
        ]
        source_states: Annotated[
            tuple[m.Cli.AtomicFileState, ...],
            m.Field(
                exclude=True,
                description="Exact source states that produced rendered content",
            ),
        ] = ()
        owner: Annotated[
            str,
            m.Field(description="Canonical artifact owner, empty for scaffold files"),
        ] = ""
        policy: Annotated[
            Literal["full", "merge", "create-only", "delegated", "manual"] | None,
            m.Field(description="Governed root artifact policy"),
        ] = None
        current_sha256: Annotated[
            str, m.Field(description="SHA-256 of current content, empty when missing")
        ] = ""
        executable: Annotated[
            bool | None,
            m.Field(description="Required executable state when mode is governed"),
        ] = None
        changed: Annotated[bool, m.Field(description="Whether content differs")]
        absent: Annotated[
            bool,
            m.Field(
                description=(
                    "When true, apply removes the path instead of writing rendered"
                )
            ),
        ] = False
        blocked: Annotated[
            bool, m.Field(description="Whether unrecognized WIP blocks application")
        ] = False
        reason: Annotated[str, m.Field(description="Blocking explanation")] = ""

    class CodegenPlan(_ConfigContract):
        """Fully validated plan produced before any managed-file write."""

        request: Annotated[
            FlextInfraConfigModels.CodegenConformRequest,
            m.Field(description="Validated public request"),
        ]
        repositories: Annotated[
            tuple[FlextInfraConfigModels.RepositoryRef, ...],
            m.Field(description="Selected repositories in deterministic order"),
        ]
        workspace: Annotated[
            FlextInfraConfigModels.WorkspaceSpec,
            m.Field(description="Workspace governing the selection"),
        ]
        make_spec: Annotated[
            FlextInfraConfigModels.MakeSpec,
            m.Field(description="Canonical Make contract"),
        ]
        uv_environments: Annotated[
            tuple[FlextInfraConfigModels.UvEnvironmentPlan, ...],
            m.Field(description="uv plans paired with selected repositories"),
        ]
        branch_ancestry: Annotated[
            tuple[FlextInfraConfigModels.BranchAncestryPlan, ...],
            m.Field(description="Governed branch ancestry observations"),
        ]
        files: Annotated[
            tuple[FlextInfraConfigModels.CodegenFilePlan, ...],
            m.Field(description="All render results validated before application"),
        ]

    class CodegenResult(_ConfigContract):
        """Public conformance outcome for check and apply modes."""

        plan: Annotated[
            FlextInfraConfigModels.CodegenPlan,
            m.Field(description="Plan that governed the operation"),
        ]
        written_files: Annotated[
            tuple[Path, ...], m.Field(description="Files atomically replaced by apply")
        ] = ()
        errors: Annotated[
            tuple[str, ...],
            m.Field(description="Fail-closed validation or write errors"),
        ] = ()


__all__: list[str] = ["FlextInfraConfigModels"]
