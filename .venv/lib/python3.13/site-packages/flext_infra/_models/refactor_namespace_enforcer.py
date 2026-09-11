"""Domain models for namespace enforcer violations and reports."""

from __future__ import annotations

from typing import Annotated, Self

from flext_core import m
from flext_infra import t
from flext_infra._models.mixins import FlextInfraModelsMixins as mm


class FlextInfraModelsNamespaceEnforcer:
    """Namespace enforcer violation and report models."""

    class FileLineViolation(mm.FileLineViolationMixin, m.ContractModel):
        """Shared base: file + line for all violation models."""

    class ImportViolationBase(mm.CurrentImportMixin, FileLineViolation):
        """Shared base: file + line + current_import."""

    class FacadeStatus(m.ContractModel):
        """Facade status."""

        family: Annotated[t.NonEmptyStr, m.Field(description="Facade family name")]
        exists: Annotated[bool, m.Field(description="Whether facade exists")]
        class_name: Annotated[str, m.Field(description="Facade class name")] = ""
        file: Annotated[str, m.Field(description="Facade file path")] = ""
        symbol_count: Annotated[
            t.NonNegativeInt, m.Field(description="Symbol count")
        ] = 0

    class LooseObjectViolation(FileLineViolation):
        """Loose object violation."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Symbol name")]
        kind: Annotated[str, m.Field(description="Object kind")]
        suggestion: Annotated[str, m.Field(description="Fix suggestion")] = ""

    class LooseTestFunctionViolation(FileLineViolation):
        """A module-level ``test_*`` function outside a ``Tests*`` class."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Test function name")]
        suggestion: Annotated[str, m.Field(description="Fix suggestion")] = ""

    class ImportAliasViolation(ImportViolationBase):
        """Import alias violation."""

        suggested_import: Annotated[
            str, m.Field(description="Suggested import statement")
        ]

    class NamespaceSourceViolation(FileLineViolation):
        """Namespace source violation."""

        alias: Annotated[t.NonEmptyStr, m.Field(description="Runtime alias letter")]
        current_source: Annotated[
            t.NonEmptyStr, m.Field(description="Current import source")
        ]
        correct_source: Annotated[
            t.NonEmptyStr, m.Field(description="Correct import source")
        ]
        current_import: Annotated[str, m.Field(description="Current import statement")]
        suggested_import: Annotated[
            str, m.Field(description="Suggested import statement")
        ]

    class ClassPlacementViolation(FileLineViolation):
        """Class placement violation."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Class name")]
        base_class: Annotated[t.NonEmptyStr, m.Field(description="Base class name")]
        suggestion: Annotated[str, m.Field(description="Fix suggestion")]
        action: Annotated[
            str, m.Field(description="Recommended fix action identifier")
        ] = "manual"
        fixable: Annotated[
            bool, m.Field(description="Whether the violation can be auto-fixed")
        ] = False
        target_facade: Annotated[
            str, m.Field(description="Target facade class suggestion")
        ] = ""
        family: Annotated[
            str, m.Field(description="Canonical family letter (c/m/p/t/u)")
        ] = ""

    class InternalImportViolation(mm.ViolationDetailMixin, ImportViolationBase):
        """Internal import violation."""

    class PrivateImportBypassViolation(mm.ViolationDetailMixin, ImportViolationBase):
        """Private-module import that should use the canonical facade."""

        private_module: Annotated[
            t.NonEmptyStr,
            m.Field(description="Fully-qualified private module being imported"),
        ]
        imported_symbol: Annotated[
            t.NonEmptyStr,
            m.Field(description="Symbol imported from the private module"),
        ]
        suggested_facade: Annotated[
            t.NonEmptyStr, m.Field(description="Canonical facade module to import from")
        ]
        symbol_exported: Annotated[
            bool,
            m.Field(description="Whether the symbol is already exported by the facade"),
        ] = False

    class InlineImportViolation(mm.ViolationDetailMixin, ImportViolationBase):
        """Inline or lazy import declared inside a function body."""

        module_name: Annotated[
            str,
            m.Field(description="Imported module name (empty for importlib dynamic)"),
        ] = ""
        imported_symbols: Annotated[
            t.StrSequence,
            m.Field(description="Symbols imported from the module, if any"),
        ] = m.Field(default_factory=tuple)
        is_importlib: Annotated[
            bool, m.Field(description="Whether this is an importlib.import_module call")
        ] = False

    class SilentFailureViolation(FileLineViolation):
        """Exception-handling construct that silences failures."""

        kind: Annotated[
            str,
            m.Field(description="Violation kind (suppress/except_pass/broad_except)"),
        ]
        detail: Annotated[
            str, m.Field(description="Human-readable violation description")
        ] = ""
        fix_action: Annotated[
            str, m.Field(description="Recommended fix action identifier")
        ] = "manual"

    class ManualProtocolViolation(FileLineViolation):
        """Manual protocol violation."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Protocol class name")]
        suggestion: Annotated[str, m.Field(description="Fix suggestion")] = (
            "Move to protocols.py/protocols/*.py/_protocols.py"
        )

    class CyclicImportViolation(m.ContractModel):
        """Cyclic import violation."""

        cycle: Annotated[
            t.VariadicTuple[str], m.Field(description="Import cycle chain")
        ]
        files: Annotated[
            t.VariadicTuple[str], m.Field(description="Files in cycle")
        ] = m.Field(default_factory=tuple)

    class RuntimeAliasViolation(
        mm.FilePathMixin,
        mm.NonNegativeLineMixin,
        mm.ViolationDetailMixin,
        m.ContractModel,
    ):
        """Runtime alias violation."""

        kind: Annotated[str, m.Field(description="Violation kind")]
        alias: Annotated[str, m.Field(description="Alias involved")]

    class FutureAnnotationsViolation(mm.FilePathMixin, m.ContractModel):
        """Future annotations violation."""

    class ManualTypingAliasViolation(mm.ViolationDetailMixin, FileLineViolation):
        """Manual typing alias violation."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Alias name")]

    class PatternSmellViolation(FileLineViolation):
        """Generic rope-detected pattern smell (ENFORCE-026..033)."""

        kind: Annotated[
            t.NonEmptyStr,
            m.Field(description="Pattern smell kind (e.g. bare_except, print)"),
        ]
        detail: Annotated[str, m.Field(description="Human-readable description")] = ""

    class CompatibilityAliasViolation(FileLineViolation):
        """Compatibility alias violation."""

        alias_name: Annotated[t.NonEmptyStr, m.Field(description="Alias name")]
        target_name: Annotated[t.NonEmptyStr, m.Field(description="Target name")]
        module_name: Annotated[
            str, m.Field(description="Source module for import-kind violations")
        ] = ""

    class ParseFailureViolation(mm.FilePathMixin, mm.ErrorDetailMixin, m.ContractModel):
        """Parse failure violation."""

        stage: Annotated[t.NonEmptyStr, m.Field(description="Parse stage")]
        error_type: Annotated[t.NonEmptyStr, m.Field(description="Error type")]

    class ProjectEnforcementReport(mm.ProjectNameMixin, m.ArbitraryTypesModel):
        """Project enforcement report."""

        project_root: Annotated[str, m.Field(description="Project root path")]
        facade_statuses: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.FacadeStatus],
            m.Field(
                default_factory=tuple,
                description="Facade status entries collected for the project.",
            ),
        ]
        loose_objects: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.LooseObjectViolation],
            m.Field(
                default_factory=tuple,
                description="Loose object violations collected for the project.",
            ),
        ]
        import_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.ImportAliasViolation],
            m.Field(
                default_factory=tuple,
                description="Import alias violations collected for the project.",
            ),
        ]
        namespace_source_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.NamespaceSourceViolation],
            m.Field(
                default_factory=tuple,
                description="Namespace source violations collected for the project.",
            ),
        ]
        internal_import_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.InternalImportViolation],
            m.Field(
                default_factory=tuple,
                description="Internal import violations collected for the project.",
            ),
        ]
        private_import_bypass_violations: Annotated[
            t.SequenceOf[
                FlextInfraModelsNamespaceEnforcer.PrivateImportBypassViolation
            ],
            m.Field(
                default_factory=tuple,
                description="Private-import bypass violations collected for the project.",
            ),
        ]
        manual_protocol_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.ManualProtocolViolation],
            m.Field(
                default_factory=tuple,
                description="Manual protocol violations collected for the project.",
            ),
        ]
        cyclic_imports: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.CyclicImportViolation],
            m.Field(
                default_factory=tuple,
                description="Cyclic import violations collected for the project.",
            ),
        ]
        runtime_alias_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.RuntimeAliasViolation],
            m.Field(
                default_factory=tuple,
                description="Runtime alias violations collected for the project.",
            ),
        ]
        future_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.FutureAnnotationsViolation],
            m.Field(
                default_factory=tuple,
                description="Future-annotations violations collected for the project.",
            ),
        ]
        manual_typing_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.ManualTypingAliasViolation],
            m.Field(
                default_factory=tuple,
                description="Manual typing alias violations collected for the project.",
            ),
        ]
        compatibility_alias_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.CompatibilityAliasViolation],
            m.Field(
                default_factory=tuple,
                description="Compatibility alias violations collected for the project.",
            ),
        ]
        foreign_canonical_alias_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.CompatibilityAliasViolation],
            m.Field(
                default_factory=tuple,
                description=(
                    "Foreign canonical alias imports collected for the project."
                ),
            ),
        ]
        class_placement_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.ClassPlacementViolation],
            m.Field(
                default_factory=tuple,
                description="Class placement violations collected for the project.",
            ),
        ]
        bare_except_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.PatternSmellViolation],
            m.Field(
                default_factory=tuple,
                description="Bare `except:` violations collected for the project.",
            ),
        ]
        print_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.PatternSmellViolation],
            m.Field(
                default_factory=tuple,
                description="`u.Cli.print()` violations collected for the project.",
            ),
        ]
        breakpoint_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.PatternSmellViolation],
            m.Field(
                default_factory=tuple,
                description="`breakpoint()` / pdb violations collected for the project.",
            ),
        ]
        open_encoding_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.PatternSmellViolation],
            m.Field(
                default_factory=tuple,
                description="`open()` without encoding violations collected for the project.",
            ),
        ]
        dict_annotation_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.PatternSmellViolation],
            m.Field(
                default_factory=tuple,
                description="`dict` annotation violations collected for the project.",
            ),
        ]
        typing_dict_attr_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.PatternSmellViolation],
            m.Field(
                default_factory=tuple,
                description="`typing.Dict` attribute violations collected for the project.",
            ),
        ]
        typing_dict_import_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.PatternSmellViolation],
            m.Field(
                default_factory=tuple,
                description="`from typing import Dict` violations collected for the project.",
            ),
        ]
        hardcoded_version_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.PatternSmellViolation],
            m.Field(
                default_factory=tuple,
                description="Hardcoded `__version__` violations collected for the project.",
            ),
        ]
        type_ignore_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.PatternSmellViolation],
            m.Field(
                default_factory=tuple,
                description="`# type: ignore` suppression violations collected for the project.",
            ),
        ]
        noqa_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.PatternSmellViolation],
            m.Field(
                default_factory=tuple,
                description="`# noqa` suppression violations collected for the project.",
            ),
        ]
        inline_import_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.InlineImportViolation],
            m.Field(
                description="Inline/lazy import violations collected for the project."
            ),
        ] = ()
        silent_failure_violations: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.SilentFailureViolation],
            m.Field(description="Silent-failure violations collected for the project."),
        ] = ()
        parse_failures: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.ParseFailureViolation],
            m.Field(
                default_factory=tuple,
                description="Parse failures collected for the project.",
            ),
        ]
        files_scanned: Annotated[
            t.NonNegativeInt, m.Field(description="Files scanned")
        ] = 0

        @m.computed_field
        @property
        def has_violations(self) -> bool:
            """Whether this project has any violations."""
            missing_facades = any(not f.exists for f in self.facade_statuses)
            violation_fields = (
                self.loose_objects,
                self.import_violations,
                self.namespace_source_violations,
                self.internal_import_violations,
                self.private_import_bypass_violations,
                self.manual_protocol_violations,
                self.cyclic_imports,
                self.runtime_alias_violations,
                self.future_violations,
                self.manual_typing_violations,
                self.compatibility_alias_violations,
                self.foreign_canonical_alias_violations,
                self.class_placement_violations,
                self.bare_except_violations,
                self.print_violations,
                self.breakpoint_violations,
                self.open_encoding_violations,
                self.dict_annotation_violations,
                self.typing_dict_attr_violations,
                self.typing_dict_import_violations,
                self.hardcoded_version_violations,
                self.type_ignore_violations,
                self.noqa_violations,
                self.inline_import_violations,
                self.silent_failure_violations,
                self.parse_failures,
            )
            return missing_facades or any(v for v in violation_fields)

    class WorkspaceEnforcementReport(m.ArbitraryTypesModel):
        """Workspace enforcement report."""

        workspace: Annotated[t.NonEmptyStr, m.Field(description="Repository root path")]
        projects: Annotated[
            t.SequenceOf[FlextInfraModelsNamespaceEnforcer.ProjectEnforcementReport],
            m.Field(
                default_factory=tuple,
                description="Per-project enforcement reports for the workspace.",
            ),
        ]
        total_facades_missing: Annotated[
            t.NonNegativeInt, m.Field(description="Total missing facades")
        ] = 0
        total_loose_objects: Annotated[
            t.NonNegativeInt, m.Field(description="Total loose objects")
        ] = 0
        total_import_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Total import violations")
        ] = 0
        total_namespace_source_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Total namespace source violations")
        ] = 0
        total_internal_import_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Total internal import violations")
        ] = 0
        total_private_import_bypass_violations: Annotated[
            t.NonNegativeInt,
            m.Field(description="Total private-import bypass violations"),
        ] = 0
        total_manual_protocol_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Total manual protocol violations")
        ] = 0
        total_cyclic_imports: Annotated[
            t.NonNegativeInt, m.Field(description="Total cyclic imports")
        ] = 0
        total_runtime_alias_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Total runtime alias violations")
        ] = 0
        total_future_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Total future annotations violations")
        ] = 0
        total_manual_typing_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Total manual typing violations")
        ] = 0
        total_compatibility_alias_violations: Annotated[
            t.NonNegativeInt,
            m.Field(description="Total compatibility alias violations"),
        ] = 0
        total_foreign_canonical_alias_violations: Annotated[
            t.NonNegativeInt,
            m.Field(description="Total foreign canonical alias import violations"),
        ] = 0
        total_class_placement_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Total class placement violations")
        ] = 0
        total_bare_except_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Total bare `except:` violations")
        ] = 0
        total_print_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Total `u.Cli.print()` violations")
        ] = 0
        total_breakpoint_violations: Annotated[
            t.NonNegativeInt,
            m.Field(description="Total `breakpoint()` / pdb violations"),
        ] = 0
        total_open_encoding_violations: Annotated[
            t.NonNegativeInt,
            m.Field(description="Total `open()` without encoding violations"),
        ] = 0
        total_dict_annotation_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Total `dict` annotation violations")
        ] = 0
        total_typing_dict_attr_violations: Annotated[
            t.NonNegativeInt,
            m.Field(description="Total `typing.Dict` attribute violations"),
        ] = 0
        total_typing_dict_import_violations: Annotated[
            t.NonNegativeInt,
            m.Field(description="Total `from typing import Dict` violations"),
        ] = 0
        total_hardcoded_version_violations: Annotated[
            t.NonNegativeInt,
            m.Field(description="Total hardcoded `__version__` violations"),
        ] = 0
        total_type_ignore_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Total `# type: ignore` violations")
        ] = 0
        total_noqa_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Total `# noqa` violations")
        ] = 0
        total_inline_import_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Total inline/lazy import violations")
        ] = 0
        total_silent_failure_violations: Annotated[
            t.NonNegativeInt, m.Field(description="Total silent-failure violations")
        ] = 0
        total_parse_failures: Annotated[
            t.NonNegativeInt, m.Field(description="Total parse failures")
        ] = 0
        total_files_scanned: Annotated[
            t.NonNegativeInt, m.Field(description="Total files scanned")
        ] = 0

        @classmethod
        def from_projects(
            cls,
            *,
            workspace: str,
            projects: t.SequenceOf[
                FlextInfraModelsNamespaceEnforcer.ProjectEnforcementReport
            ],
        ) -> Self:
            """From projects."""
            return cls(
                workspace=workspace,
                projects=projects,
                total_facades_missing=sum(
                    1 for p in projects for f in p.facade_statuses if not f.exists
                ),
                total_loose_objects=sum(len(p.loose_objects) for p in projects),
                total_import_violations=sum(len(p.import_violations) for p in projects),
                total_namespace_source_violations=sum(
                    len(p.namespace_source_violations) for p in projects
                ),
                total_internal_import_violations=sum(
                    len(p.internal_import_violations) for p in projects
                ),
                total_private_import_bypass_violations=sum(
                    len(p.private_import_bypass_violations) for p in projects
                ),
                total_manual_protocol_violations=sum(
                    len(p.manual_protocol_violations) for p in projects
                ),
                total_cyclic_imports=sum(len(p.cyclic_imports) for p in projects),
                total_runtime_alias_violations=sum(
                    len(p.runtime_alias_violations) for p in projects
                ),
                total_future_violations=sum(len(p.future_violations) for p in projects),
                total_manual_typing_violations=sum(
                    len(p.manual_typing_violations) for p in projects
                ),
                total_compatibility_alias_violations=sum(
                    len(p.compatibility_alias_violations) for p in projects
                ),
                total_foreign_canonical_alias_violations=sum(
                    len(p.foreign_canonical_alias_violations) for p in projects
                ),
                total_class_placement_violations=sum(
                    len(p.class_placement_violations) for p in projects
                ),
                total_bare_except_violations=sum(
                    len(p.bare_except_violations) for p in projects
                ),
                total_print_violations=sum(len(p.print_violations) for p in projects),
                total_breakpoint_violations=sum(
                    len(p.breakpoint_violations) for p in projects
                ),
                total_open_encoding_violations=sum(
                    len(p.open_encoding_violations) for p in projects
                ),
                total_dict_annotation_violations=sum(
                    len(p.dict_annotation_violations) for p in projects
                ),
                total_typing_dict_attr_violations=sum(
                    len(p.typing_dict_attr_violations) for p in projects
                ),
                total_typing_dict_import_violations=sum(
                    len(p.typing_dict_import_violations) for p in projects
                ),
                total_hardcoded_version_violations=sum(
                    len(p.hardcoded_version_violations) for p in projects
                ),
                total_type_ignore_violations=sum(
                    len(p.type_ignore_violations) for p in projects
                ),
                total_noqa_violations=sum(len(p.noqa_violations) for p in projects),
                total_inline_import_violations=sum(
                    len(p.inline_import_violations) for p in projects
                ),
                total_silent_failure_violations=sum(
                    len(p.silent_failure_violations) for p in projects
                ),
                total_parse_failures=sum(len(p.parse_failures) for p in projects),
                total_files_scanned=sum(p.files_scanned for p in projects),
            )

        @m.computed_field
        @property
        def has_violations(self) -> bool:
            """Has violations."""
            return any(
                getattr(self, field_name) > 0
                for field_name in type(self).model_fields
                if field_name.startswith("total_")
                and field_name != "total_files_scanned"
            )


__all__: list[str] = ["FlextInfraModelsNamespaceEnforcer"]
