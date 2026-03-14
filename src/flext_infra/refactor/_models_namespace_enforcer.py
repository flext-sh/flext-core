from __future__ import annotations

from typing import Annotated, Self

from pydantic import ConfigDict, Field

from flext_core import FlextModels


class NamespaceFacadeStatus(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    family: Annotated[str, Field(min_length=1)]
    exists: Annotated[bool, Field()]
    class_name: Annotated[str, Field(default="")]
    file: Annotated[str, Field(default="")]
    symbol_count: Annotated[int, Field(default=0, ge=0)]

    @classmethod
    def create(
        cls,
        *,
        family: str,
        exists: bool,
        class_name: str,
        file: str,
        symbol_count: int,
    ) -> Self:
        return cls(
            family=family,
            exists=exists,
            class_name=class_name,
            file=file,
            symbol_count=symbol_count,
        )


class NamespaceLooseObjectViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: Annotated[str, Field(min_length=1)]
    line: Annotated[int, Field(ge=1)]
    name: Annotated[str, Field(min_length=1)]
    kind: Annotated[str, Field()]
    suggestion: Annotated[str, Field(default="")]

    @classmethod
    def create(
        cls,
        *,
        file: str,
        line: int,
        name: str,
        kind: str,
        suggestion: str,
    ) -> Self:
        return cls(
            file=file,
            line=line,
            name=name,
            kind=kind,
            suggestion=suggestion,
        )


class NamespaceImportAliasViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: Annotated[str, Field(min_length=1)]
    line: Annotated[int, Field(ge=1)]
    current_import: Annotated[str, Field()]
    suggested_import: Annotated[str, Field()]

    @classmethod
    def create(
        cls,
        *,
        file: str,
        line: int,
        current_import: str,
        suggested_import: str,
    ) -> Self:
        return cls(
            file=file,
            line=line,
            current_import=current_import,
            suggested_import=suggested_import,
        )


class NamespaceInternalImportViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: Annotated[str, Field(min_length=1)]
    line: Annotated[int, Field(ge=1)]
    current_import: Annotated[str, Field()]
    detail: Annotated[str, Field()]

    @classmethod
    def create(cls, *, file: str, line: int, current_import: str, detail: str) -> Self:
        return cls(
            file=file,
            line=line,
            current_import=current_import,
            detail=detail,
        )


class NamespaceManualProtocolViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: Annotated[str, Field(min_length=1)]
    line: Annotated[int, Field(ge=1)]
    name: Annotated[str, Field(min_length=1)]
    suggestion: Annotated[
        str, Field(default="Move to protocols.py/protocols/*.py/_protocols.py")
    ] = "Move to protocols.py/protocols/*.py/_protocols.py"

    @classmethod
    def create(cls, *, file: str, line: int, name: str, suggestion: str = "") -> Self:
        if len(suggestion) > 0:
            return cls(file=file, line=line, name=name, suggestion=suggestion)
        return cls(
            file=file,
            line=line,
            name=name,
            suggestion="Move to protocols.py/protocols/*.py/_protocols.py",
        )


class NamespaceCyclicImportViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    cycle: Annotated[tuple[str, ...], Field()]
    files: Annotated[tuple[str, ...], Field(default_factory=tuple)]

    @classmethod
    def create(cls, *, cycle: tuple[str, ...], files: tuple[str, ...]) -> Self:
        return cls(cycle=cycle, files=files)


class NamespaceRuntimeAliasViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: Annotated[str, Field(min_length=1)]
    line: Annotated[int, Field(default=0, ge=0)]
    kind: Annotated[str, Field()]
    alias: Annotated[str, Field()]
    detail: Annotated[str, Field(default="")]

    @classmethod
    def create(
        cls,
        *,
        file: str,
        kind: str,
        alias: str,
        detail: str,
        line: int = 0,
    ) -> Self:
        return cls(
            file=file,
            line=line,
            kind=kind,
            alias=alias,
            detail=detail,
        )


class NamespaceFutureAnnotationsViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: Annotated[str, Field(min_length=1)]

    @classmethod
    def create(cls, *, file: str) -> Self:
        return cls(file=file)


class NamespaceManualTypingAliasViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: Annotated[str, Field(min_length=1)]
    line: Annotated[int, Field(ge=1)]
    name: Annotated[str, Field(min_length=1)]
    detail: Annotated[str, Field(default="")]

    @classmethod
    def create(cls, *, file: str, line: int, name: str, detail: str) -> Self:
        return cls(
            file=file,
            line=line,
            name=name,
            detail=detail,
        )


class NamespaceCompatibilityAliasViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: Annotated[str, Field(min_length=1)]
    line: Annotated[int, Field(ge=1)]
    alias_name: Annotated[str, Field(min_length=1)]
    target_name: Annotated[str, Field(min_length=1)]

    @classmethod
    def create(cls, *, file: str, line: int, alias_name: str, target_name: str) -> Self:
        return cls(
            file=file,
            line=line,
            alias_name=alias_name,
            target_name=target_name,
        )


class NamespaceParseFailureViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: Annotated[str, Field(min_length=1)]
    stage: Annotated[str, Field(min_length=1)]
    error_type: Annotated[str, Field(min_length=1)]
    detail: Annotated[str, Field(default="")]

    @classmethod
    def create(cls, *, file: str, stage: str, error_type: str, detail: str) -> Self:
        return cls(
            file=file,
            stage=stage,
            error_type=error_type,
            detail=detail,
        )


class NamespaceProjectEnforcementReport(FlextModels.ArbitraryTypesModel):
    project: Annotated[str, Field(min_length=1)]
    project_root: Annotated[str, Field()]
    facade_statuses: Annotated[
        list[NamespaceFacadeStatus],
        Field(
            default_factory=list,
        ),
    ]
    loose_objects: Annotated[
        list[NamespaceLooseObjectViolation],
        Field(
            default_factory=list,
        ),
    ]
    import_violations: Annotated[
        list[NamespaceImportAliasViolation],
        Field(
            default_factory=list,
        ),
    ]
    internal_import_violations: Annotated[
        list[NamespaceInternalImportViolation],
        Field(
            default_factory=list,
        ),
    ]
    manual_protocol_violations: Annotated[
        list[NamespaceManualProtocolViolation],
        Field(
            default_factory=list,
        ),
    ]
    cyclic_imports: Annotated[
        list[NamespaceCyclicImportViolation],
        Field(
            default_factory=list,
        ),
    ]
    runtime_alias_violations: Annotated[
        list[NamespaceRuntimeAliasViolation],
        Field(
            default_factory=list,
        ),
    ]
    future_violations: Annotated[
        list[NamespaceFutureAnnotationsViolation],
        Field(
            default_factory=list,
        ),
    ]
    manual_typing_violations: Annotated[
        list[NamespaceManualTypingAliasViolation],
        Field(
            default_factory=list,
        ),
    ]
    compatibility_alias_violations: Annotated[
        list[NamespaceCompatibilityAliasViolation],
        Field(
            default_factory=list,
        ),
    ]
    parse_failures: Annotated[
        list[NamespaceParseFailureViolation],
        Field(
            default_factory=list,
        ),
    ]
    files_scanned: Annotated[int, Field(default=0, ge=0)]

    @classmethod
    def create(
        cls,
        *,
        project: str,
        project_root: str,
        facade_statuses: list[NamespaceFacadeStatus],
        loose_objects: list[NamespaceLooseObjectViolation],
        import_violations: list[NamespaceImportAliasViolation],
        internal_import_violations: list[NamespaceInternalImportViolation],
        manual_protocol_violations: list[NamespaceManualProtocolViolation],
        cyclic_imports: list[NamespaceCyclicImportViolation],
        runtime_alias_violations: list[NamespaceRuntimeAliasViolation],
        future_violations: list[NamespaceFutureAnnotationsViolation],
        manual_typing_violations: list[NamespaceManualTypingAliasViolation],
        compatibility_alias_violations: list[NamespaceCompatibilityAliasViolation],
        parse_failures: list[NamespaceParseFailureViolation],
        files_scanned: int,
    ) -> Self:
        return cls(
            project=project,
            project_root=project_root,
            facade_statuses=facade_statuses,
            loose_objects=loose_objects,
            import_violations=import_violations,
            internal_import_violations=internal_import_violations,
            manual_protocol_violations=manual_protocol_violations,
            cyclic_imports=cyclic_imports,
            runtime_alias_violations=runtime_alias_violations,
            future_violations=future_violations,
            manual_typing_violations=manual_typing_violations,
            compatibility_alias_violations=compatibility_alias_violations,
            parse_failures=parse_failures,
            files_scanned=files_scanned,
        )


def _empty_project_reports() -> list[NamespaceProjectEnforcementReport]:
    return []


class NamespaceWorkspaceEnforcementReport(FlextModels.ArbitraryTypesModel):
    workspace: Annotated[str, Field(min_length=1)]
    projects: Annotated[
        list[NamespaceProjectEnforcementReport],
        Field(
            default_factory=_empty_project_reports,
        ),
    ] = Field(default_factory=_empty_project_reports)
    total_facades_missing: Annotated[int, Field(default=0, ge=0)]
    total_loose_objects: Annotated[int, Field(default=0, ge=0)]
    total_import_violations: Annotated[int, Field(default=0, ge=0)]
    total_internal_import_violations: Annotated[int, Field(default=0, ge=0)]
    total_manual_protocol_violations: Annotated[int, Field(default=0, ge=0)]
    total_cyclic_imports: Annotated[int, Field(default=0, ge=0)]
    total_runtime_alias_violations: Annotated[int, Field(default=0, ge=0)]
    total_future_violations: Annotated[int, Field(default=0, ge=0)]
    total_manual_typing_violations: Annotated[int, Field(default=0, ge=0)]
    total_compatibility_alias_violations: Annotated[int, Field(default=0, ge=0)]
    total_parse_failures: Annotated[int, Field(default=0, ge=0)]
    total_files_scanned: Annotated[int, Field(default=0, ge=0)]

    @classmethod
    def create(
        cls,
        *,
        workspace: str,
        projects: list[NamespaceProjectEnforcementReport],
        total_facades_missing: int,
        total_loose_objects: int,
        total_import_violations: int,
        total_internal_import_violations: int,
        total_manual_protocol_violations: int,
        total_cyclic_imports: int,
        total_runtime_alias_violations: int,
        total_future_violations: int,
        total_manual_typing_violations: int,
        total_compatibility_alias_violations: int,
        total_parse_failures: int,
        total_files_scanned: int,
    ) -> Self:
        return cls(
            workspace=workspace,
            projects=projects,
            total_facades_missing=total_facades_missing,
            total_loose_objects=total_loose_objects,
            total_import_violations=total_import_violations,
            total_internal_import_violations=total_internal_import_violations,
            total_manual_protocol_violations=total_manual_protocol_violations,
            total_cyclic_imports=total_cyclic_imports,
            total_runtime_alias_violations=total_runtime_alias_violations,
            total_future_violations=total_future_violations,
            total_manual_typing_violations=total_manual_typing_violations,
            total_compatibility_alias_violations=total_compatibility_alias_violations,
            total_parse_failures=total_parse_failures,
            total_files_scanned=total_files_scanned,
        )

    @property
    def has_violations(self) -> bool:
        """Check if any violations exist across the workspace."""
        return (
            self.total_facades_missing > 0
            or self.total_loose_objects > 0
            or self.total_import_violations > 0
            or self.total_internal_import_violations > 0
            or self.total_manual_protocol_violations > 0
            or self.total_cyclic_imports > 0
            or self.total_runtime_alias_violations > 0
            or self.total_future_violations > 0
            or self.total_manual_typing_violations > 0
            or self.total_compatibility_alias_violations > 0
            or self.total_parse_failures > 0
        )


class FlextInfraNamespaceEnforcerModels:
    FacadeStatus = NamespaceFacadeStatus
    LooseObjectViolation = NamespaceLooseObjectViolation
    ImportAliasViolation = NamespaceImportAliasViolation
    InternalImportViolation = NamespaceInternalImportViolation
    ManualProtocolViolation = NamespaceManualProtocolViolation
    CyclicImportViolation = NamespaceCyclicImportViolation
    RuntimeAliasViolation = NamespaceRuntimeAliasViolation
    FutureAnnotationsViolation = NamespaceFutureAnnotationsViolation
    ManualTypingAliasViolation = NamespaceManualTypingAliasViolation
    CompatibilityAliasViolation = NamespaceCompatibilityAliasViolation
    ParseFailureViolation = NamespaceParseFailureViolation
    NamespaceEnforcementReport = NamespaceProjectEnforcementReport
    ProjectEnforcementReport = NamespaceProjectEnforcementReport
    ProjectEnforcementWorkspaceReport = NamespaceWorkspaceEnforcementReport
    WorkspaceEnforcementReport = NamespaceWorkspaceEnforcementReport


__all__ = ["FlextInfraNamespaceEnforcerModels"]
