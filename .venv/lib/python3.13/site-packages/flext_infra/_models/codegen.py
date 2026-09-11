"""Domain models for the codegen subpackage."""

from __future__ import annotations

from collections.abc import MutableSet
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self

from flext_cli import m, u
from flext_infra import c, p, t
from flext_infra._models._defaults import ImmutableEmptyMapping
from flext_infra._models.codegen_render import FlextInfraModelsCodegenRender
from flext_infra._models.mixins import FlextInfraModelsMixins as mm


class FlextInfraModelsCodegen(FlextInfraModelsCodegenRender):
    """Models for codegen census, scaffold, and auto-fix pipelines."""

    class MiseToolchainArtifactPaths(m.ArbitraryTypesModel):
        """Canonical live toolchain-bundle destinations for one project."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True, extra="forbid")

        config: Annotated[
            Path, m.Field(description="Generated Mise configuration destination")
        ]
        unix_launcher: Annotated[Path, m.Field(description="Unix launcher destination")]
        windows_launcher: Annotated[
            Path, m.Field(description="Windows launcher destination")
        ]
        lock: Annotated[Path, m.Field(description="Project Mise lock destination")]

    class MiseToolchainProjectLayout(m.ArbitraryTypesModel):
        """Stable paths needed to validate and recover one project."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True, extra="forbid")

        selector: Annotated[
            t.NonEmptyStr, m.Field(description="Workspace-relative project selector")
        ]
        root: Annotated[Path, m.Field(description="Resolved project root")]
        transaction_root: Annotated[
            Path,
            m.Field(
                description="Persistent transaction root on this project filesystem"
            ),
        ]
        artifacts: Annotated[
            FlextInfraModelsCodegen.MiseToolchainArtifactPaths,
            m.Field(description="Canonical artifact destinations"),
        ]

    class MiseToolchainWorkspaceLayout(m.ArbitraryTypesModel):
        """Stable recovery topology independent of mutable source contents."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True, extra="forbid")

        scope_root: Annotated[Path, m.Field(description="Resolved transaction scope")]
        state_root: Annotated[
            Path, m.Field(description="Persistent common lock and journal directory")
        ]
        projects: Annotated[
            tuple[FlextInfraModelsCodegen.MiseToolchainProjectLayout, ...],
            m.Field(min_length=1, description="Ordered complete workspace topology"),
        ]

    class MiseToolchainConfigState(m.ArbitraryTypesModel):
        """Current destination plus the exact planned Mise configuration."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True, extra="forbid")

        before: Annotated[
            m.Cli.AtomicFileState,
            m.Field(description="Exact preflight state of the live configuration"),
        ]
        replacement_content: Annotated[
            bytes,
            m.Field(
                min_length=1,
                strict=True,
                description="Exact rendered bytes to stage and publish",
            ),
        ]
        replacement_mode: Annotated[
            int,
            m.Field(
                ge=0,
                le=0o7777,
                strict=True,
                description="Exact permission mode to stage and publish",
            ),
        ]
        sources: Annotated[
            tuple[m.Cli.AtomicFileState, ...],
            m.Field(description="Ordered YAML states that produced the replacement"),
        ] = ()

    class MiseToolchainProjectState(m.ArbitraryTypesModel):
        """Immutable source and destination snapshot for one project layout."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True, extra="forbid")

        layout: Annotated[
            FlextInfraModelsCodegen.MiseToolchainProjectLayout,
            m.Field(description="Stable project layout owning this snapshot"),
        ]
        config: Annotated[
            FlextInfraModelsCodegen.MiseToolchainConfigState,
            m.Field(description="Planned generated Mise configuration state"),
        ]
        artifacts: Annotated[
            FlextInfraModelsCodegen.MiseToolchainArtifactSet,
            m.Field(description="Named launcher and lock states"),
        ]

        @u.model_validator(mode="after")
        def _validate_destination_paths(self) -> Self:
            """Bind every captured state to its declared live destination."""
            expected = (
                self.layout.artifacts.config,
                self.layout.artifacts.unix_launcher,
                self.layout.artifacts.windows_launcher,
                self.layout.artifacts.lock,
            )
            observed = (
                self.config.before.path,
                self.artifacts.unix_launcher.path,
                self.artifacts.windows_launcher.path,
                self.artifacts.lock.path,
            )
            if observed != expected:
                msg = "Mise project states differ from declared destinations"
                raise ValueError(msg)
            return self

    class MiseToolchainArtifactSet(m.ArbitraryTypesModel):
        """Named file states that prevent artifact-order ambiguity."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True, extra="forbid")

        unix_launcher: Annotated[
            m.Cli.AtomicFileState, m.Field(description="Observed Unix launcher state")
        ]
        windows_launcher: Annotated[
            m.Cli.AtomicFileState,
            m.Field(description="Observed Windows launcher state"),
        ]
        lock: Annotated[
            m.Cli.AtomicFileState,
            m.Field(description="Observed project Mise lock state"),
        ]

    class MiseToolchainWorkspacePlan(m.ArbitraryTypesModel):
        """One stable layout plus a coherent mutable-state snapshot."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True, extra="forbid")

        layout: Annotated[
            FlextInfraModelsCodegen.MiseToolchainWorkspaceLayout,
            m.Field(description="Stable workspace topology"),
        ]
        projects: Annotated[
            tuple[FlextInfraModelsCodegen.MiseToolchainProjectState, ...],
            m.Field(min_length=1, description="Ordered complete workspace topology"),
        ]

        @u.model_validator(mode="after")
        def _validate_project_layouts(self) -> Self:
            """Bind every mutable project snapshot to the exact stable layout."""
            if (
                tuple(project.layout for project in self.projects)
                != self.layout.projects
            ):
                msg = "Mise project snapshots differ from workspace layout"
                raise ValueError(msg)
            return self

    class MiseToolchainPublication(m.ArbitraryTypesModel):
        """One guarded replacement belonging to the current Mise transaction."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True, extra="forbid")

        before: Annotated[
            m.Cli.AtomicFileState,
            m.Field(description="Exact destination state observed during preflight"),
        ]
        replacement: Annotated[
            m.Cli.AtomicFileState,
            m.Field(description="Exact caller-owned staged replacement state"),
        ]

        @u.model_validator(mode="after")
        def _validate_replacement_present(self) -> Self:
            """Require every publication to own a materialized staged file."""
            if self.replacement.content is None or self.replacement.mode is None:
                msg = "Mise publication replacement must be present"
                raise ValueError(msg)
            return self

    class MiseToolchainJournalSource(m.ArbitraryTypesModel):
        """One immutable source identity guarded by a Mise transaction journal."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True, extra="forbid")

        path: Annotated[
            t.NonEmptyStr,
            m.Field(description="Workspace-relative committed source path"),
        ]
        sha256: Annotated[
            str,
            m.Field(
                pattern=r"^[0-9a-f]{64}$",
                description="Exact source-byte SHA-256 identity",
            ),
        ]
        mode: Annotated[
            int,
            m.Field(
                ge=0, le=0o7777, strict=True, description="Exact source permission bits"
            ),
        ]

    class MiseToolchainJournalEntry(m.ArbitraryTypesModel):
        """Recoverable before/after identity for one published Mise artifact."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True, extra="forbid")

        path: Annotated[
            t.NonEmptyStr,
            m.Field(description="Workspace-relative live artifact destination"),
        ]
        original_exists: Annotated[
            bool,
            m.Field(
                strict=True, description="Whether the destination existed at preflight"
            ),
        ]
        original_backup: Annotated[
            t.NonEmptyStr | None,
            m.Field(description="Workspace-relative original-byte backup"),
        ] = None
        original_sha256: Annotated[
            str | None,
            m.Field(
                pattern=r"^[0-9a-f]{64}$",
                description="Exact original-byte SHA-256 identity when present",
            ),
        ] = None
        original_mode: Annotated[
            int | None,
            m.Field(
                ge=0,
                le=0o7777,
                strict=True,
                description="Exact original permission bits when present",
            ),
        ] = None
        replacement_sha256: Annotated[
            str,
            m.Field(
                pattern=r"^[0-9a-f]{64}$",
                description="Exact replacement-byte SHA-256 identity",
            ),
        ]
        replacement_mode: Annotated[
            int,
            m.Field(
                ge=0, le=0o7777, strict=True, description="Replacement permission bits"
            ),
        ]

        @u.model_validator(mode="after")
        def _validate_original_tuple(self) -> Self:
            """Require complete recovery identity exactly when original existed."""
            original = (self.original_backup, self.original_sha256, self.original_mode)
            populated = tuple(value is not None for value in original)
            if (self.original_exists and not all(populated)) or (
                not self.original_exists and any(populated)
            ):
                msg = "Mise journal original recovery tuple is inconsistent"
                raise ValueError(msg)
            return self

    class MiseToolchainRecoveryAction(m.ArbitraryTypesModel):
        """One preclassified recovery decision with no live effect applied."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True, extra="forbid")

        entry: Annotated[
            FlextInfraModelsCodegen.MiseToolchainJournalEntry,
            m.Field(description="Journal entry owning the recovery decision"),
        ]
        current: Annotated[
            m.Cli.AtomicFileState,
            m.Field(description="Exact live target state used for classification"),
        ]
        operation: Annotated[
            Literal["noop", "delete", "restore"],
            m.Field(description="Only authorized recovery effect for the target"),
        ]

    class MiseToolchainJournal(m.ArbitraryTypesModel):
        """Persisted recovery contract for one workspace-wide Mise publication."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True, extra="forbid")

        version: Annotated[
            Literal[3], m.Field(description="Exact journal schema version")
        ]
        state: Annotated[
            Literal["staging", "prepared", "committed"],
            m.Field(description="Durable publication transition state"),
        ]
        projects: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                min_length=1,
                description="Ordered project selectors owned by this transaction",
            ),
        ]
        sources: Annotated[
            tuple[FlextInfraModelsCodegen.MiseToolchainJournalSource, ...],
            m.Field(description="Source identities used by staging"),
        ]
        entries: Annotated[
            tuple[FlextInfraModelsCodegen.MiseToolchainJournalEntry, ...],
            m.Field(description="Recoverable artifact transitions"),
        ]

        @u.model_validator(mode="after")
        def _validate_lifecycle(self) -> Self:
            """Bind staging and publication payloads to one safe project set."""
            if self.projects[0] != "." and "." in self.projects:
                msg = "Mise root selector must be first when present"
                raise ValueError(msg)
            if len(set(self.projects)) != len(self.projects):
                msg = "Mise journal project selectors must be unique"
                raise ValueError(msg)
            for selector in self.projects:
                relative = Path(selector)
                if (
                    relative.is_absolute()
                    or relative.as_posix() != selector
                    or ".." in relative.parts
                ):
                    msg = f"unsafe Mise journal project selector: {selector}"
                    raise ValueError(msg)
            if self.state == "staging" and self.entries:
                msg = "staging Mise journal must not authorize live transitions"
                raise ValueError(msg)
            if self.state != "staging" and not self.entries:
                msg = "published Mise journal must contain recovery entries"
                raise ValueError(msg)
            return self

    class CensusViolation(mm.RequiredNonNegativeLineMixin, m.ArbitraryTypesModel):
        """A single namespace violation detected by the census service."""

        module: t.NonEmptyStr = m.Field(description="Module file path")
        rule: t.NonEmptyStr = m.Field(
            description="Violated rule identifier (e.g. NS-001)"
        )
        message: t.NonEmptyStr = m.Field(description="Human-readable violation message")
        fixable: bool = m.Field(description="Whether this violation can be auto-fixed")

    class CensusReport(mm.ProjectNameMixin, m.ArbitraryTypesModel):
        """Aggregated census report for a single project."""

        @staticmethod
        def _violations_default() -> list[FlextInfraModelsCodegen.CensusViolation]:
            """Violations default."""
            return []

        violations: Annotated[
            list[FlextInfraModelsCodegen.CensusViolation],
            m.Field(
                default_factory=_violations_default, description="Detected violations"
            ),
        ]
        total: Annotated[t.NonNegativeInt, m.Field(description="Total violation count")]
        fixable: Annotated[
            t.NonNegativeInt, m.Field(description="Count of auto-fixable violations")
        ]

    class ScaffoldResult(mm.ProjectNameMixin, m.ArbitraryTypesModel):
        """Result of scaffolding base modules for a project.

        Enforcement exemption: internal tooling model with intentional
        mutable state.
        """

        files_created: t.StrSequence = m.Field(
            default_factory=tuple, description="Newly created file paths"
        )
        files_skipped: t.StrSequence = m.Field(
            default_factory=tuple, description="Skipped (already existing) file paths"
        )

    class ScaffoldDirRequest(m.ArbitraryTypesModel):
        """Directory-level scaffold request and accumulation state."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            arbitrary_types_allowed=True, revalidate_instances="never"
        )

        target_dir: Annotated[Path, m.Field(description="Directory to scaffold")]
        prefix: Annotated[str, m.Field(description="Generated class name prefix")]
        modules: Annotated[
            t.VariadicTuple[t.Quad[str, str, str, str]],
            m.Field(description="Module skeleton definitions"),
        ]
        test_prefix: Annotated[str, m.Field(description="Generated test class prefix")]
        base_module: Annotated[
            t.NonEmptyStr,
            m.Field(description="Explicit module owning every generated base class"),
        ]
        dry_run: Annotated[
            bool, m.Field(description="Whether to report creations without writing")
        ]
        files_created: Annotated[
            t.MutableSequenceOf[str], m.Field(description="Created file accumulator")
        ]
        files_skipped: Annotated[
            t.MutableSequenceOf[str], m.Field(description="Skipped file accumulator")
        ]

    class AutoFixResult(mm.ProjectNameMixin, m.ArbitraryTypesModel):
        """Result of auto-fixing namespace violations for a project."""

        @staticmethod
        def _violations_default() -> list[FlextInfraModelsCodegen.CensusViolation]:
            """Violations default."""
            return []

        violations_fixed: Annotated[
            list[FlextInfraModelsCodegen.CensusViolation],
            m.Field(
                default_factory=_violations_default, description="Fixed violations"
            ),
        ]
        violations_skipped: Annotated[
            list[FlextInfraModelsCodegen.CensusViolation],
            m.Field(
                default_factory=_violations_default,
                description="Skipped violations (not auto-fixable)",
            ),
        ]
        files_modified: t.StrSequence = m.Field(
            default_factory=tuple, description="Modified file paths"
        )

    class ConsolidatorFileResult(m.ContractModel):
        """Per-file result emitted by the constants consolidator."""

        file: Annotated[
            t.NonEmptyStr, m.Field(description="Workspace-relative file path")
        ]
        status: Annotated[
            Literal["applied", "reverted"],
            m.Field(description="File processing status"),
        ]
        changes: Annotated[
            t.StrSequence,
            m.Field(default_factory=tuple, description="Applied replacements"),
        ]

    class ConsolidatorReport(m.ContractModel):
        """JSON report emitted by the constants consolidator."""

        total_found: Annotated[
            t.NonNegativeInt, m.Field(description="Total replacements found")
        ] = 0
        total_applied: Annotated[
            t.NonNegativeInt, m.Field(description="Total replacements applied")
        ] = 0
        total_failed: Annotated[
            t.NonNegativeInt, m.Field(description="Total files reverted")
        ] = 0
        files: Annotated[
            t.SequenceOf[FlextInfraModelsCodegen.ConsolidatorFileResult],
            m.Field(default_factory=tuple, description="Per-file processing results"),
        ]

    class NamespaceModulePolicy(m.ArbitraryTypesModel):
        """Derived gen-init policy for one governed module."""

        enforce_contract: Annotated[
            bool, m.Field(description="Whether gen-init must enforce namespace shape.")
        ] = False
        export_symbols: Annotated[
            bool,
            m.Field(description="Whether gen-init should discover public symbols."),
        ] = False
        include_in_lazy_init: Annotated[
            bool,
            m.Field(description="Whether lazy-init should index this module at all."),
        ] = True
        project_prefix: Annotated[
            str, m.Field(description="Canonical class prefix expected for the module.")
        ] = ""
        expected_alias: Annotated[
            str | None,
            m.Field(description="Canonical module-level alias allowed for the file."),
        ] = None
        expected_family: Annotated[
            str | None,
            m.Field(description="Canonical namespace family suffix for the file."),
        ] = None
        family_tokens: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Accepted family markers for private namespace modules.",
        )
        accepted_suffixes: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Accepted class suffixes for governed facade classes.",
        )
        allow_main_export: Annotated[
            bool,
            m.Field(description="Whether the file may export a module-level main()."),
        ] = False
        allow_type_alias: Annotated[
            bool,
            m.Field(description="Whether the module may keep TypeAlias declarations."),
        ] = False
        is_fixture_module: Annotated[
            bool,
            m.Field(
                description="Whether the module belongs to a private fixtures package."
            ),
        ] = False
        type_checking_imports: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Canonical root names allowed inside TYPE_CHECKING imports.",
        )

    class LazyInitPackageContext(m.ArbitraryTypesModel):
        """Declarative package context for one lazy-init directory."""

        pkg_dir: Path = m.Field(description="Directory being processed.")
        init_path: Path = m.Field(description="Target __init__.py path.")
        current_pkg: str = m.Field(description="Importable package name.")
        surface: str = m.Field(description="Root surface for wrapper alias resolution.")
        generated_init: Annotated[
            bool,
            m.Field(
                description="Whether the current __init__.py is generated by lazy-init."
            ),
        ] = False
        importable: Annotated[
            bool,
            m.Field(
                description="Whether the directory resolves to an importable package."
            ),
        ] = False

    class LazyInitPlan(m.ArbitraryTypesModel):
        """Fully resolved lazy-init action and render payload.

        Enforcement exemption: internal tooling model with intentional
        mutable state.
        """

        context: FlextInfraModelsCodegen.LazyInitPackageContext = m.Field(
            description="Discovered package context."
        )
        action: Annotated[
            c.Infra.LazyInitAction,
            m.Field(description="Action selected for this package."),
        ] = c.Infra.LazyInitAction.SKIP
        exports: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Public exports for generated __init__.py.",
        )
        lazy_map: t.LazyAliasMap = m.Field(
            default_factory=ImmutableEmptyMapping,
            description="Lazy import map: export name to module/attribute target.",
        )
        type_checking_map: t.LazyAliasMap = m.Field(
            default_factory=ImmutableEmptyMapping,
            description=(
                "Type-checking import map used to publish static package attributes "
                "without widening the runtime/public lazy export surface."
            ),
        )
        eager_dunders: t.LazyAliasMap = m.Field(
            default_factory=ImmutableEmptyMapping,
            description=(
                "Dunder exports that must be eagerly imported at __init__.py "
                "load time. Required for the ``__version__.py`` submodule case "
                "where the submodule name collides with the dunder string it "
                "exports — eager binding pins the canonical string in the "
                "package dict before any submodule re-import can shadow it."
            ),
        )
        inline_constants: t.StrMapping = m.Field(
            default_factory=ImmutableEmptyMapping,
            description="Inline constants emitted directly into __init__.py.",
        )
        wildcard_runtime_modules: t.StrSequence = m.Field(
            default_factory=tuple, description="Runtime wildcard import modules."
        )
        child_packages_for_lazy: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Direct child package imports merged at runtime.",
        )
        excluded_lazy_names: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Names excluded from runtime child lazy import merges.",
        )

    class QualityGateCheck(m.ArbitraryTypesModel):
        """A single quality gate check result entry."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Check identifier")]
        passed: Annotated[bool, m.Field(description="Whether check passed")]
        detail: Annotated[str, m.Field(description="Human-readable check detail")] = ""
        critical: Annotated[bool, m.Field(description="Whether failure is critical")]

    class QualityGateProjectFinding(mm.ProjectNameMixin, m.ArbitraryTypesModel):
        """Per-project quality gate findings."""

        violations_total: Annotated[
            t.NonNegativeInt, m.Field(description="Total violations")
        ]
        fixable_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Auto-fixable violations")
        ]
        validator_passed: Annotated[
            bool, m.Field(description="Whether validator passed")
        ]
        flext_failures: Annotated[
            t.NonNegativeInt, m.Field(description="FLEXT failure count")
        ]
        layer_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Layer violation count")
        ]
        cross_project_reference_violations: Annotated[
            t.NonNegativeInt,
            m.Field(description="Cross-project reference violation count"),
        ]

    class BulkFixItem(
        mm.AbsoluteFilePathTextMixin, mm.PositiveLineMixin, m.ArbitraryTypesModel
    ):
        """Shared line-addressable item used by bulk codegen fixes."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Item identifier")]

    class ConstantDefinition(mm.ProjectNameMixin, mm.NestedClassPathMixin, BulkFixItem):
        """A single constant extracted from a constants.py file."""

        value_repr: Annotated[
            str, m.Field(description="Source repr (e.g., '30', '\"localhost\"')")
        ]
        type_annotation: Annotated[
            str, m.Field(description="Type annotation string")
        ] = ""

    class DuplicateConstantGroup(m.ArbitraryTypesModel):
        """Cross-project duplicate group with consolidation metadata."""

        constant_name: t.NonEmptyStr = m.Field(description="Constant identifier")
        definitions: list[FlextInfraModelsCodegen.ConstantDefinition] = m.Field(
            description="Definitions across projects"
        )
        is_value_identical: bool = m.Field(description="Whether all values match")
        canonical_ref: Annotated[
            str, m.Field(description="Canonical parent reference")
        ] = ""

    class DirectConstantRef(mm.ProjectNameMixin, m.ArbitraryTypesModel):
        """Direct FlextXConstants.Y.Z reference that should use c.* alias."""

        full_ref: Annotated[
            t.NonEmptyStr,
            m.Field(description="e.g., FlextAuthConstants.Auth.DEFAULT_TIMEOUT"),
        ]
        alias_ref: Annotated[
            t.NonEmptyStr, m.Field(description="e.g., c.Auth.DEFAULT_TIMEOUT")
        ]
        file_path: Annotated[
            t.NonEmptyStr, m.Field(description="File containing the reference")
        ]
        line: Annotated[t.PositiveInt, m.Field(description="Line number")]

    class CanonicalValueRule(m.ArbitraryTypesModel):
        """Canonical value rule."""

        value: Annotated[t.Infra.CanonicalValue, m.Field(description="Canonical value")]
        type: Annotated[
            Literal["int", "str", "regex", "frozenset", "tuple"],
            m.Field(description="Canonical type"),
        ]
        canonical_ref: str = m.Field(description="Canonical reference")
        semantic_names: t.StrSequence = m.Field(
            default_factory=tuple, description="semantic_names"
        )

        @u.model_validator(mode="after")
        def validate_value_shape(self) -> Self:
            """Keep canonical governance values aligned with their declared kind."""
            if self.type == "int":
                if not isinstance(self.value, int) or isinstance(self.value, bool):
                    msg = "int canonical values must use an integer payload"
                    raise TypeError(msg)
                return self
            if self.type in {"str", "regex"}:
                if not isinstance(self.value, str):
                    msg = "string canonical values must use a string payload"
                    raise TypeError(msg)
                return self
            if isinstance(self.value, str):
                msg = "sequence canonical values must use a string sequence payload"
                raise TypeError(msg)
            return self

    class NsRule(m.ArbitraryTypesModel):
        """Ns rule."""

        id: str = m.Field(description="Rule ID")
        description: str = m.Field(description="Rule description")
        fixable: bool = m.Field(description="Whether the rule is fixable")
        fixable_exclusion: Annotated[
            str | None, m.Field(description="Fixable exclusion reason")
        ] = None

    class ConstantsGovernanceConfig(m.ArbitraryTypesModel):
        """Constants governance config."""

        version: str = m.Field(description="Config version")
        rules: list[FlextInfraModelsCodegen.NsRule] = m.Field(
            description="Governance rules"
        )
        canonical_values: list[FlextInfraModelsCodegen.CanonicalValueRule] = m.Field(
            description="Canonical values settings"
        )
        constants_class_pattern: str = m.Field(
            description="Constants class pattern regex"
        )

    class TestTreeRulesConfig(m.ArbitraryTypesModel):
        """Config-driven parameters for the loose-test-function detector.

        Loaded from ``rules/test-tree-rules.yml`` (business rule = config SSOT);
        the detector is a pure engine over these values (never hardcoded).
        """

        model_config = m.ConfigDict(frozen=True, extra="forbid")

        version: str = m.Field(description="Config version")
        test_dir_globs: tuple[str, ...] = m.Field(
            description="Globs (relative to project root) whose files are tests"
        )
        test_fn_prefix: str = m.Field(
            description="Prefix marking a module-level function as a test"
        )
        required_class_prefix: str = m.Field(
            description="Class prefix a test function must be nested under"
        )

    class TestImportDagRulesConfig(m.ArbitraryTypesModel):
        """Validated policy for the strict package-test import DAG."""

        model_config = m.ConfigDict(frozen=True, extra="forbid")

        version: str = m.Field(description="Config version")
        facet_order: tuple[str, ...] = m.Field(
            description="Allowed directed order for canonical test facets"
        )
        facet_files: t.MappingKV[str, str] = m.Field(
            description="Canonical test facet module filenames"
        )
        fixture_parts: tuple[str, ...] = m.Field(
            description="Test infrastructure path or module parts"
        )
        test_module_prefix: str = m.Field(
            description="Prefix identifying collected test modules"
        )
        shared_package: str = m.Field(
            description="Package owning shared test infrastructure"
        )
        shared_allowed_imports: tuple[str, ...] = m.Field(
            description="Packages shared test infrastructure may import"
        )

    class FixContext(m.ArbitraryTypesModel):
        """Mutable accumulation context for fix operations.

        Enforcement exemption: MutableSequence/MutableSet accumulators are
        appended/added to as fixes proceed; fresh per-instance — no shared
        state.
        """

        @staticmethod
        def _violations_default() -> list[FlextInfraModelsCodegen.CensusViolation]:
            """Violations default."""
            return []

        violations_fixed: Annotated[
            t.MutableSequenceOf[FlextInfraModelsCodegen.CensusViolation],
            m.Field(
                default_factory=_violations_default,
                description="List of violations that were fixed",
            ),
        ] = m.Field(
            default_factory=_violations_default,
            description="List of violations that were fixed",
        )
        violations_skipped: Annotated[
            t.MutableSequenceOf[FlextInfraModelsCodegen.CensusViolation],
            m.Field(
                default_factory=_violations_default,
                description="List of violations that were skipped",
            ),
        ] = m.Field(
            default_factory=_violations_default,
            description="List of violations that were skipped",
        )
        files_modified: Annotated[
            MutableSet[str],
            m.Field(
                default_factory=set, description="Set of unique modified file paths"
            ),
        ] = m.Field(
            default_factory=set, description="Set of unique modified file paths"
        )

        @property
        def has_changes(self) -> bool:
            """Whether at least one file was modified."""
            return bool(self.files_modified)

        def skip(self, *, module: str, rule: str, line: int, message: str) -> None:
            """Skip."""
            self.violations_skipped.append(
                FlextInfraModelsCodegen.CensusViolation(
                    module=module, rule=rule, line=line, message=message, fixable=False
                )
            )

        def fix(self, *, module: str, rule: str, line: int, message: str) -> None:
            """Fix."""
            self.violations_fixed.append(
                FlextInfraModelsCodegen.CensusViolation(
                    module=module, rule=rule, line=line, message=message, fixable=True
                )
            )

    class ViolationKey(m.ContractModel):
        """Content-stable violation identifier — resilient to line shifts."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True, extra="forbid")

        module: Annotated[str, m.Field(description="Module containing the violation")]
        rule: Annotated[str, m.Field(description="Rule that was violated")]
        content_hash: Annotated[
            str, m.Field(description="SHA256 of surrounding context lines")
        ]

        def __hash__(self) -> int:
            """Hash by stable business identity so keys work in sets and frozensets."""
            return hash((self.module, self.rule, self.content_hash))

        @staticmethod
        def from_violation(
            violation: FlextInfraModelsCodegen.CensusViolation,
            source_lines: t.StrSequence,
        ) -> FlextInfraModelsCodegen.ViolationKey:
            """Build key from violation and source context (+-2 lines)."""
            ctx_start = max(0, violation.line - 2)
            ctx_end = min(len(source_lines), violation.line + 3)
            context = "\n".join(source_lines[ctx_start:ctx_end])
            content_hash = u.Cli.sha256_content(context)
            return FlextInfraModelsCodegen.ViolationKey(
                module=violation.module, rule=violation.rule, content_hash=content_hash
            )

    class CodegenPipelineState(m.ArbitraryTypesModel):
        """Typed inter-stage state for the codegen pipeline — Pydantic v2 model."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            extra="forbid", arbitrary_types_allowed=True
        )

        discovered_projects: Annotated[
            t.SequenceOf[p.Infra.ProjectInfo],
            m.Field(description="Projects discovered at pipeline start"),
        ] = ()
        census_service: Annotated[
            p.Infra.CodegenCensusService | None,
            m.Field(description="Cached census service for reuse across stages"),
        ] = None
        reports_before: Annotated[
            t.SequenceOf[FlextInfraModelsCodegen.CensusReport],
            m.Field(description="Census reports collected before fixes"),
        ] = ()
        reports_after: Annotated[
            t.SequenceOf[FlextInfraModelsCodegen.CensusReport],
            m.Field(description="Census reports collected after fixes"),
        ] = ()
        scaffold_results: Annotated[
            t.SequenceOf[FlextInfraModelsCodegen.ScaffoldResult],
            m.Field(description="Scaffolding stage results"),
        ] = ()
        fix_results: Annotated[
            t.SequenceOf[FlextInfraModelsCodegen.AutoFixResult],
            m.Field(description="Auto-fix stage results"),
        ] = ()


__all__: list[str] = ["FlextInfraModelsCodegen"]
