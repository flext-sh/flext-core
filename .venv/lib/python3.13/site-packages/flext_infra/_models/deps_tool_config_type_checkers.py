"""Pyright and Pyrefly tool configuration models for the deps subpackage."""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import m
from flext_infra import c, t
from flext_infra._models._defaults import ImmutableEmptyMapping


class FlextInfraModelsDepsToolConfigTypeCheckers:
    """Type checker configuration models."""

    class PyrightConfig(m.ArbitraryTypesModel):
        """Pyright strict settings loaded from YAML."""

        class DiagnosticPathOverride(m.ContractModel):
            """One evidence-backed diagnostic override for an existing path."""

            root: Annotated[
                t.NonEmptyStr, m.Field(description="Project-relative override root.")
            ]
            report_private_usage: Annotated[
                str,
                m.Field(
                    alias=c.Infra.REPORT_PRIVATE_USAGE,
                    description="Narrow reportPrivateUsage value for this root.",
                ),
            ]
            rationale: Annotated[
                t.NonEmptyStr,
                m.Field(description="Verified technical reason for the override."),
            ]

        class ExecutionEnvironment(m.ContractModel):
            """Pyright execution environment entry."""

            model_config: ClassVar[m.ConfigDict] = m.ConfigDict(populate_by_name=True)

            root: Annotated[
                str, m.Field(description="Execution environment root path.")
            ]
            report_private_usage: Annotated[
                str,
                m.Field(
                    alias=c.Infra.REPORT_PRIVATE_USAGE,
                    description="reportPrivateUsage override for this environment.",
                ),
            ]
            extra_paths: Annotated[
                t.StrSequence,
                m.Field(
                    alias=c.Infra.EXTRA_PATHS,
                    description="extraPaths applied to this execution environment.",
                ),
            ]
            rationale: Annotated[
                str,
                m.Field(
                    exclude=True,
                    description="Non-rendered evidence for a scoped environment.",
                ),
            ] = ""

        class PathRulesConfig(m.ArbitraryTypesModel):
            """Path resolution rules loaded from YAML."""

            source_dir: Annotated[
                str,
                m.Field(
                    alias="source-dir", description="Primary source directory name."
                ),
            ]
            project_root: Annotated[
                str,
                m.Field(
                    alias="project-root",
                    description="Project root path entry used in path resolution.",
                ),
            ]
            env_dirs: Annotated[
                t.StrSequence,
                m.Field(
                    alias="env-dirs",
                    description="Canonical execution environment directories in order.",
                ),
            ]
            test_like_dirs: Annotated[
                t.StrSequence,
                m.Field(
                    alias="test-like-dirs",
                    description="Env dirs that should include project root in extraPaths.",
                ),
            ]
            default_excludes: Annotated[
                t.StrSequence,
                m.Field(
                    alias="default-excludes",
                    description="Always-applied pyright exclude globs.",
                ),
            ]
            root_typings_paths: Annotated[
                t.StrSequence,
                m.Field(
                    alias="root-typings-paths",
                    description="Typings paths used in workspace settings.",
                ),
            ]
            project_typings_paths: Annotated[
                t.StrSequence,
                m.Field(
                    alias="project-typings-paths",
                    description="Typings paths used in declared_repository configs.",
                ),
            ]
            ignored_diagnostic_globs: Annotated[
                t.StrSequence,
                m.Field(
                    alias="ignored-diagnostic-globs",
                    description="Pyright ignore globs for files that remain on search paths but must not emit diagnostics.",
                ),
            ]
            diagnostic_path_overrides: Annotated[
                tuple[
                    FlextInfraModelsDepsToolConfigTypeCheckers.PyrightConfig.DiagnosticPathOverride,
                    ...,
                ],
                m.Field(
                    alias="diagnostic-path-overrides",
                    description="Existing path roots with evidence-backed diagnostic overrides.",
                ),
            ] = ()
            source_report_private_usage: Annotated[
                str,
                m.Field(
                    alias="source-report-private-usage",
                    description="reportPrivateUsage value for source-dir execution envs.",
                ),
            ]
            test_like_report_private_usage: Annotated[
                str,
                m.Field(
                    alias="test-like-report-private-usage",
                    description="reportPrivateUsage value for test-like execution envs.",
                ),
            ]
            other_report_private_usage: Annotated[
                str,
                m.Field(
                    alias="other-report-private-usage",
                    description="reportPrivateUsage value for non-source/non-test-like envs.",
                ),
            ]
            venv_name: Annotated[
                str,
                m.Field(
                    alias="venv-name",
                    description="Virtualenv directory name shared across pyright configs.",
                ),
            ]

        strict_settings: Annotated[
            t.StrMapping,
            m.Field(
                alias="strict-settings", description="Pyright strict baseline options."
            ),
        ]
        extended_settings: Annotated[
            t.StrMapping,
            m.Field(
                alias="extended-settings",
                description="Pyright extended settings options.",
            ),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        lazy_import_suppressions: Annotated[
            t.StrMapping,
            m.Field(
                alias="lazy-import-suppressions",
                description="Pyright rules suppressed in ALL envs due to lazy import pattern.",
            ),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        global_suppression_rationales: Annotated[
            t.StrMapping,
            m.Field(
                alias="global-suppression-rationales",
                description="Global Pyright exclusions mapped to verified facade-FLEXT rationales.",
            ),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        source_env_suppressions: Annotated[
            t.StrMapping,
            m.Field(
                alias="source-env-suppressions",
                description="Additional pyright rules suppressed in source env only.",
            ),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        test_like_env_suppressions: Annotated[
            t.StrMapping,
            m.Field(
                alias="test-like-env-suppressions",
                description="Additional pyright rules suppressed in test-like envs.",
            ),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        path_rules: Annotated[
            FlextInfraModelsDepsToolConfigTypeCheckers.PyrightConfig.PathRulesConfig,
            m.Field(
                alias="path-rules",
                description="Pyright path resolution and execution-env rules.",
            ),
        ]

    class PyreflyConfig(m.ArbitraryTypesModel):
        """Pyrefly strict settings loaded from YAML."""

        class PathRulesConfig(m.ArbitraryTypesModel):
            """Path resolution rules loaded from YAML."""

            source_dir: Annotated[
                str,
                m.Field(
                    alias="source-dir", description="Primary source directory name."
                ),
            ]
            project_root: Annotated[
                str,
                m.Field(
                    alias="project-root",
                    description="Project root path entry used in search-path.",
                ),
            ]
            root_typings_paths: Annotated[
                t.StrSequence,
                m.Field(
                    alias="root-typings-paths",
                    description="Typings paths used in workspace settings.",
                ),
            ]
            project_typings_paths: Annotated[
                t.StrSequence,
                m.Field(
                    alias="project-typings-paths",
                    description="Typings paths used in declared_repository configs.",
                ),
            ]
            env_dirs: Annotated[
                t.StrSequence,
                m.Field(
                    alias="env-dirs",
                    description="Canonical directories used to build project-includes.",
                ),
            ]
            workspace_include_children: Annotated[
                bool,
                m.Field(
                    alias="workspace-include-children",
                    description="Whether root pyrefly should include child projects.",
                ),
            ]
            workspace_include_child_env_dirs: Annotated[
                t.StrSequence,
                m.Field(
                    alias="workspace-include-child-env-dirs",
                    description="Child env dirs included by root pyrefly when enabled.",
                ),
            ]
            project_shared_search_paths: Annotated[
                t.StrSequence,
                m.Field(
                    alias="project-shared-search-paths",
                    description="Additional shared workspace paths for declared_repository pyrefly search-path.",
                ),
            ]

        python_version: Annotated[
            str,
            m.Field(
                alias="python-version", description="Pyrefly python-version baseline."
            ),
        ]
        disable_project_excludes_heuristics: Annotated[
            bool,
            m.Field(
                alias="disable-project-excludes-heuristics",
                description="Disable implicit project-exclude heuristics in pyrefly.",
            ),
        ]
        use_ignore_files: Annotated[
            bool,
            m.Field(
                alias="use-ignore-files",
                description="Whether pyrefly should apply .gitignore/.ignore files.",
            ),
        ]
        strict_errors: Annotated[
            t.StrSequence,
            m.Field(
                alias="strict-errors",
                description="Pyrefly errors enabled as strict defaults.",
            ),
        ]
        project_exclude_globs: Annotated[
            t.StrSequence,
            m.Field(
                alias="project-exclude-globs",
                description="Globs excluded from pyrefly project checking.",
            ),
        ]
        path_rules: Annotated[
            FlextInfraModelsDepsToolConfigTypeCheckers.PyreflyConfig.PathRulesConfig,
            m.Field(alias="path-rules", description="Pyrefly search-path rules."),
        ]


__all__: list[str] = ["FlextInfraModelsDepsToolConfigTypeCheckers"]
