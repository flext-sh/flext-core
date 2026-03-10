from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field

from flext_core import FlextModels


def _empty_facade_statuses() -> list[NamespaceFacadeStatus]:
    return []


def _empty_loose_objects() -> list[NamespaceLooseObjectViolation]:
    return []


def _empty_import_violations() -> list[NamespaceImportAliasViolation]:
    return []


def _empty_internal_import_violations() -> list[NamespaceInternalImportViolation]:
    return []


def _empty_manual_protocol_violations() -> list[NamespaceManualProtocolViolation]:
    return []


def _empty_cyclic_imports() -> list[NamespaceCyclicImportViolation]:
    return []


def _empty_runtime_alias_violations() -> list[NamespaceRuntimeAliasViolation]:
    return []


def _empty_future_violations() -> list[NamespaceFutureAnnotationsViolation]:
    return []


def _empty_manual_typing_violations() -> list[NamespaceManualTypingAliasViolation]:
    return []


def _empty_compatibility_alias_violations() -> list[
    NamespaceCompatibilityAliasViolation
]:
    return []


def _empty_parse_failures() -> list[NamespaceParseFailureViolation]:
    return []


def _empty_project_reports() -> list[NamespaceProjectEnforcementReport]:
    return []


class NamespaceFacadeStatus(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    family: str = Field(min_length=1)
    exists: bool = Field()
    class_name: str = Field(default="")
    file: str = Field(default="")
    symbol_count: int = Field(default=0, ge=0)

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
        return cls.model_validate({
            "family": family,
            "exists": exists,
            "class_name": class_name,
            "file": file,
            "symbol_count": symbol_count,
        })


class NamespaceLooseObjectViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: str = Field(min_length=1)
    line: int = Field(ge=1)
    name: str = Field(min_length=1)
    kind: str = Field()
    suggestion: str = Field(default="")

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
        return cls.model_validate({
            "file": file,
            "line": line,
            "name": name,
            "kind": kind,
            "suggestion": suggestion,
        })


class NamespaceImportAliasViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: str = Field(min_length=1)
    line: int = Field(ge=1)
    current_import: str = Field()
    suggested_import: str = Field()

    @classmethod
    def create(
        cls,
        *,
        file: str,
        line: int,
        current_import: str,
        suggested_import: str,
    ) -> Self:
        return cls.model_validate({
            "file": file,
            "line": line,
            "current_import": current_import,
            "suggested_import": suggested_import,
        })


class NamespaceInternalImportViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: str = Field(min_length=1)
    line: int = Field(ge=1)
    current_import: str = Field()
    detail: str = Field()

    @classmethod
    def create(cls, *, file: str, line: int, current_import: str, detail: str) -> Self:
        return cls.model_validate({
            "file": file,
            "line": line,
            "current_import": current_import,
            "detail": detail,
        })


class NamespaceManualProtocolViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: str = Field(min_length=1)
    line: int = Field(ge=1)
    name: str = Field(min_length=1)
    suggestion: str = Field(default="Move to protocols.py/protocols/*.py/_protocols.py")

    @classmethod
    def create(cls, *, file: str, line: int, name: str, suggestion: str = "") -> Self:
        payload = {"file": file, "line": line, "name": name}
        if len(suggestion) > 0:
            payload["suggestion"] = suggestion
        return cls.model_validate(payload)


class NamespaceCyclicImportViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    cycle: tuple[str, ...] = Field()
    files: tuple[str, ...] = Field(default_factory=tuple)

    @classmethod
    def create(cls, *, cycle: tuple[str, ...], files: tuple[str, ...]) -> Self:
        return cls.model_validate({"cycle": cycle, "files": files})


class NamespaceRuntimeAliasViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: str = Field(min_length=1)
    line: int = Field(default=0, ge=0)
    kind: str = Field()
    alias: str = Field()
    detail: str = Field(default="")

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
        return cls.model_validate({
            "file": file,
            "line": line,
            "kind": kind,
            "alias": alias,
            "detail": detail,
        })


class NamespaceFutureAnnotationsViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: str = Field(min_length=1)

    @classmethod
    def create(cls, *, file: str) -> Self:
        return cls.model_validate({"file": file})


class NamespaceManualTypingAliasViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: str = Field(min_length=1)
    line: int = Field(ge=1)
    name: str = Field(min_length=1)
    detail: str = Field(default="")

    @classmethod
    def create(cls, *, file: str, line: int, name: str, detail: str) -> Self:
        return cls.model_validate({
            "file": file,
            "line": line,
            "name": name,
            "detail": detail,
        })


class NamespaceCompatibilityAliasViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: str = Field(min_length=1)
    line: int = Field(ge=1)
    alias_name: str = Field(min_length=1)
    target_name: str = Field(min_length=1)

    @classmethod
    def create(cls, *, file: str, line: int, alias_name: str, target_name: str) -> Self:
        return cls.model_validate({
            "file": file,
            "line": line,
            "alias_name": alias_name,
            "target_name": target_name,
        })


class NamespaceParseFailureViolation(FlextModels.ArbitraryTypesModel):
    model_config = ConfigDict(frozen=True)

    file: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    detail: str = Field(default="")

    @classmethod
    def create(cls, *, file: str, stage: str, error_type: str, detail: str) -> Self:
        return cls.model_validate({
            "file": file,
            "stage": stage,
            "error_type": error_type,
            "detail": detail,
        })


class NamespaceProjectEnforcementReport(FlextModels.ArbitraryTypesModel):
    project: str = Field(min_length=1)
    project_root: str = Field()
    facade_statuses: list[NamespaceFacadeStatus] = Field(
        default_factory=_empty_facade_statuses,
    )
    loose_objects: list[NamespaceLooseObjectViolation] = Field(
        default_factory=_empty_loose_objects,
    )
    import_violations: list[NamespaceImportAliasViolation] = Field(
        default_factory=_empty_import_violations,
    )
    internal_import_violations: list[NamespaceInternalImportViolation] = Field(
        default_factory=_empty_internal_import_violations,
    )
    manual_protocol_violations: list[NamespaceManualProtocolViolation] = Field(
        default_factory=_empty_manual_protocol_violations,
    )
    cyclic_imports: list[NamespaceCyclicImportViolation] = Field(
        default_factory=_empty_cyclic_imports,
    )
    runtime_alias_violations: list[NamespaceRuntimeAliasViolation] = Field(
        default_factory=_empty_runtime_alias_violations,
    )
    future_violations: list[NamespaceFutureAnnotationsViolation] = Field(
        default_factory=_empty_future_violations,
    )
    manual_typing_violations: list[NamespaceManualTypingAliasViolation] = Field(
        default_factory=_empty_manual_typing_violations,
    )
    compatibility_alias_violations: list[NamespaceCompatibilityAliasViolation] = Field(
        default_factory=_empty_compatibility_alias_violations,
    )
    parse_failures: list[NamespaceParseFailureViolation] = Field(
        default_factory=_empty_parse_failures,
    )
    files_scanned: int = Field(default=0, ge=0)

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
        return cls.model_validate({
            "project": project,
            "project_root": project_root,
            "facade_statuses": facade_statuses,
            "loose_objects": loose_objects,
            "import_violations": import_violations,
            "internal_import_violations": internal_import_violations,
            "manual_protocol_violations": manual_protocol_violations,
            "cyclic_imports": cyclic_imports,
            "runtime_alias_violations": runtime_alias_violations,
            "future_violations": future_violations,
            "manual_typing_violations": manual_typing_violations,
            "compatibility_alias_violations": compatibility_alias_violations,
            "parse_failures": parse_failures,
            "files_scanned": files_scanned,
        })


class NamespaceWorkspaceEnforcementReport(FlextModels.ArbitraryTypesModel):
    workspace: str = Field(min_length=1)
    projects: list[NamespaceProjectEnforcementReport] = Field(
        default_factory=_empty_project_reports,
    )
    total_facades_missing: int = Field(default=0, ge=0)
    total_loose_objects: int = Field(default=0, ge=0)
    total_import_violations: int = Field(default=0, ge=0)
    total_internal_import_violations: int = Field(default=0, ge=0)
    total_manual_protocol_violations: int = Field(default=0, ge=0)
    total_cyclic_imports: int = Field(default=0, ge=0)
    total_runtime_alias_violations: int = Field(default=0, ge=0)
    total_future_violations: int = Field(default=0, ge=0)
    total_manual_typing_violations: int = Field(default=0, ge=0)
    total_compatibility_alias_violations: int = Field(default=0, ge=0)
    total_parse_failures: int = Field(default=0, ge=0)
    total_files_scanned: int = Field(default=0, ge=0)

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
        return cls.model_validate({
            "workspace": workspace,
            "projects": projects,
            "total_facades_missing": total_facades_missing,
            "total_loose_objects": total_loose_objects,
            "total_import_violations": total_import_violations,
            "total_internal_import_violations": total_internal_import_violations,
            "total_manual_protocol_violations": total_manual_protocol_violations,
            "total_cyclic_imports": total_cyclic_imports,
            "total_runtime_alias_violations": total_runtime_alias_violations,
            "total_future_violations": total_future_violations,
            "total_manual_typing_violations": total_manual_typing_violations,
            "total_compatibility_alias_violations": total_compatibility_alias_violations,
            "total_parse_failures": total_parse_failures,
            "total_files_scanned": total_files_scanned,
        })

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
    class NamespaceEnforcementModels:
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
