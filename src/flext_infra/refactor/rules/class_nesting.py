"""Class nesting refactor rule: move loose classes under namespace classes."""

from __future__ import annotations

from pathlib import Path

import libcst as cst
import yaml
from libcst.metadata import MetadataWrapper
from pydantic import TypeAdapter, ValidationError

from flext_infra import FlextInfraJsonService, c, m, t
import flext_infra.refactor._utilities as refactor_utilities
from flext_infra.refactor.transformers.class_nesting import (
    FlextInfraRefactorClassNestingTransformer,
)
from flext_infra.refactor.transformers.helper_consolidation import (
    HelperConsolidationTransformer,
)
from flext_infra.refactor.transformers.nested_class_propagation import (
    NestedClassPropagationTransformer,
)
from flext_infra.refactor.validation import PostCheckGate

type _MappingEntry = dict[str, str]
type _PolicyFamily = dict[str, t.ContainerValue]
type _PolicyDocument = dict[str, t.ContainerValue]
type _RuleConfig = dict[str, str | list[_MappingEntry]]
type _PostCheckPayload = dict[str, str | list[str]]
type _PreCheckViolation = dict[str, str]


class FlextInfraRefactorClassNestingRuleUtilities:
    """Typed helper functions for class-nesting rule orchestration."""

    @staticmethod
    def entry_list(value: str | list[_MappingEntry] | None) -> list[_MappingEntry]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        msg = "class nesting entries must be a list"
        raise ValueError(msg)

    @staticmethod
    def string_list(value: t.ContainerValue | None) -> list[str]:
        if value is None:
            return []
        try:
            return TypeAdapter(list[str]).validate_python(value)
        except ValidationError:
            msg = "expected list[str] value"
            raise ValueError(msg) from None

    @staticmethod
    def mapping_list(
        value: t.ContainerValue | None,
    ) -> list[dict[str, t.ContainerValue]]:
        if value is None:
            return []
        try:
            return TypeAdapter(list[dict[str, t.ContainerValue]]).validate_python(value)
        except ValidationError:
            msg = "expected list[dict[str, ContainerValue]] value"
            raise ValueError(msg) from None


class PreCheckGate:
    """Validate mapping entries against family-level policy before transforms."""

    def __init__(self, policy_path: Path | None = None) -> None:
        """Initialize pre-check gate with optional policy path."""
        self._policy_path = policy_path or Path(__file__).with_name(
            "class-policy-v2.yml"
        )
        self._schema_path = self._policy_path.with_name("class-policy-v2.schema.json")
        self._policy_by_family = self._load_policy()

    def validate_entry(
        self, entry: _MappingEntry
    ) -> tuple[bool, _PreCheckViolation | None]:
        """Return (ok, violation) for a single mapping entry."""
        source_symbol = entry.get(c.Infra.ReportKeys.LOOSE_NAME, "")
        helper_symbol = entry.get("helper_name", "")
        symbol = source_symbol or helper_symbol
        target_namespace = entry.get(c.Infra.ReportKeys.TARGET_NAMESPACE, "")
        current_file = entry.get(c.Infra.ReportKeys.CURRENT_FILE, "")
        if not symbol or not target_namespace or not current_file:
            return True, None

        module_family = self._module_family_from_path(current_file)
        if module_family == "other_private":
            return True, None
        policy = self._policy_by_family.get(module_family)
        if policy is None:
            return False, {
                c.Infra.ReportKeys.RULE_ID: f"precheck:{symbol}",
                c.Infra.ReportKeys.SOURCE_SYMBOL: symbol,
                c.Infra.ReportKeys.VIOLATION_TYPE: "unknown_module_family",
                c.Infra.ReportKeys.SUGGESTED_FIX: f"declare explicit policy for {module_family}",
            }

        operation = (
            c.Infra.ReportKeys.HELPER_CONSOLIDATION
            if helper_symbol
            else c.Infra.ReportKeys.CLASS_NESTING
        )
        allowed_operations = _string_list(policy.get("allowed_operations"))
        if operation not in allowed_operations:
            return False, {
                c.Infra.ReportKeys.RULE_ID: f"precheck:{symbol}",
                c.Infra.ReportKeys.SOURCE_SYMBOL: symbol,
                c.Infra.ReportKeys.VIOLATION_TYPE: "operation_not_allowed",
                c.Infra.ReportKeys.SUGGESTED_FIX: f"allow {operation} in policy for {module_family}",
            }

        forbidden_operations = _string_list(policy.get("forbidden_operations"))
        if operation in forbidden_operations:
            return False, {
                c.Infra.ReportKeys.RULE_ID: f"precheck:{symbol}",
                c.Infra.ReportKeys.SOURCE_SYMBOL: symbol,
                c.Infra.ReportKeys.VIOLATION_TYPE: "operation_forbidden",
                c.Infra.ReportKeys.SUGGESTED_FIX: f"remove {operation} from forbidden_operations for {module_family}",
            }

        forbidden_targets = _string_list(
            policy.get(c.Infra.ReportKeys.FORBIDDEN_TARGETS)
        )
        if any(
            self._target_matches(target_namespace, pattern)
            for pattern in forbidden_targets
        ):
            return False, {
                c.Infra.ReportKeys.RULE_ID: f"precheck:{symbol}",
                c.Infra.ReportKeys.SOURCE_SYMBOL: symbol,
                c.Infra.ReportKeys.VIOLATION_TYPE: "forbidden_target",
                c.Infra.ReportKeys.SUGGESTED_FIX: f"choose allowed target for family {module_family}",
            }
        return True, None

    def _load_policy(self) -> dict[str, _PolicyFamily]:
        try:
            loaded_raw = yaml.safe_load(
                self._policy_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            )
        except (OSError, yaml.YAMLError):
            return {}
        try:
            loaded = TypeAdapter(dict[str, t.ContainerValue]).validate_python(
                loaded_raw
            )
        except ValidationError:
            return {}
        if not self._schema_valid(loaded):
            return {}
        policy_matrix = _mapping_list(loaded.get("policy_matrix"))

        by_family: dict[str, _PolicyFamily] = {}
        for raw in policy_matrix:
            family_name = raw.get("family_name")
            if not isinstance(family_name, str):
                continue
            forbidden_targets = _string_list(
                raw.get(c.Infra.ReportKeys.FORBIDDEN_TARGETS)
            )

            allowed_operations = _string_list(raw.get("allowed_operations"))

            forbidden_operations = _string_list(raw.get("forbidden_operations"))

            validation_requirements = raw.get("validation_requirements")
            by_family[family_name] = {
                "family_name": family_name,
                "allowed_operations": allowed_operations,
                "forbidden_operations": forbidden_operations,
                c.Infra.ReportKeys.FORBIDDEN_TARGETS: forbidden_targets,
                "validation_requirements": validation_requirements or {},
            }
        return by_family

    def _schema_valid(self, loaded: _PolicyDocument) -> bool:
        schema_raw = FlextInfraJsonService().load(self._schema_path)
        if schema_raw is None:
            return False
        try:
            schema = TypeAdapter(dict[str, t.ContainerValue]).validate_python(
                schema_raw
            )
        except ValidationError:
            return False

        top_required = _string_list(schema.get("required", []))
        if not self._has_required_fields(loaded, top_required):
            return False

        definitions_raw = schema.get("definitions", {})
        try:
            definitions = TypeAdapter(dict[str, t.ContainerValue]).validate_python(
                definitions_raw
            )
        except ValidationError:
            return False

        policy_entry_raw = definitions.get("policyEntry", {})
        class_rule_raw = definitions.get("classRule", {})
        try:
            policy_entry = TypeAdapter(dict[str, t.ContainerValue]).validate_python(
                policy_entry_raw
            )
            class_rule = TypeAdapter(dict[str, t.ContainerValue]).validate_python(
                class_rule_raw
            )
        except ValidationError:
            return False

        policy_entry_required = _string_list(policy_entry.get("required", []))
        class_rule_required = _string_list(class_rule.get("required", []))

        policy_matrix = _mapping_list(loaded.get("policy_matrix"))
        for entry in policy_matrix:
            if not self._has_required_fields(entry, policy_entry_required):
                return False

        rules = _mapping_list(loaded.get(c.Infra.ReportKeys.RULES))
        for rule in rules:
            if not self._has_required_fields(rule, class_rule_required):
                return False

        return True

    def _has_required_fields(
        self,
        entry: t.ContainerValue,
        required_fields: list[str],
    ) -> bool:
        if not isinstance(entry, dict):
            return False
        return all(key in entry for key in required_fields)

    def _module_family_from_path(self, path: str) -> str:
        normalized = path.replace("\\", "/")
        if "_models" in normalized:
            return "_models"
        if "_utilities" in normalized:
            return "_utilities"
        if "_dispatcher" in normalized:
            return "_dispatcher"
        if "_decorators" in normalized:
            return "_decorators"
        if "_runtime" in normalized:
            return "_runtime"
        return "other_private"

    def _target_matches(self, target_namespace: str, pattern: str) -> bool:
        if pattern.endswith(".*"):
            return target_namespace.lower().startswith(pattern[:-2].lower())
        return target_namespace == pattern


class ClassNestingRefactorRule:
    """Apply class-nesting transforms driven by YAML mapping files."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize rule with an optional path to the YAML config."""
        self._config_path = config_path or Path(__file__).with_name(
            "class-nesting-mappings.yml"
        )
        self._policy_path = Path(__file__).with_name("class-policy-v2.yml")
        self._pre_check_gate = PreCheckGate()
        self._post_check_gate = PostCheckGate()

    def apply(
        self, file_path: Path, *, dry_run: bool = False
    ) -> m.Infra.Refactor.Result:
        """Transform *file_path* according to loaded mappings and policy."""
        try:
            if file_path.suffix != c.Infra.Extensions.PYTHON:
                return m.Infra.Refactor.Result(
                    file_path=file_path,
                    success=True,
                    modified=False,
                    changes=["Skipped non-Python file"],
                    refactored_code=None,
                )

            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = cst.parse_module(source)
            mappings = self._load_config()
            confidence_threshold = self._confidence_threshold(mappings)

            class_mappings = self._class_nesting_mappings(
                mappings,
                file_path,
                confidence_threshold,
            )
            helper_mappings = self._helper_consolidation_mappings(
                mappings,
                file_path,
                confidence_threshold,
            )
            class_renames = self._class_rename_mappings(
                mappings,
                file_path,
                confidence_threshold,
            )
            policy_context = self._policy_context_from_document()
            class_families = self._families_for_scope(
                entries=self._entries_for_source_file(
                    FlextInfraRefactorClassNestingRuleUtilities.entry_list(
                        mappings.get(c.Infra.ReportKeys.CLASS_NESTING)
                    ),
                    file_path,
                    confidence_threshold,
                ),
                symbol_key=c.Infra.ReportKeys.LOOSE_NAME,
            )
            helper_families = self._families_for_scope(
                entries=self._entries_for_source_file(
                    FlextInfraRefactorClassNestingRuleUtilities.entry_list(
                        mappings.get(c.Infra.ReportKeys.HELPER_CONSOLIDATION)
                    ),
                    file_path,
                    confidence_threshold,
                ),
                symbol_key="helper_name",
            )

            precheck_violations = self._run_precheck(
                mappings,
                file_path,
                confidence_threshold,
            )
            if precheck_violations:
                return m.Infra.Refactor.Result(
                    file_path=file_path,
                    success=False,
                    modified=False,
                    error="precheck_failed",
                    changes=precheck_violations,
                    refactored_code=None,
                )

            changes: list[str] = []
            tree = self._apply_class_nesting(
                tree,
                class_mappings,
                changes,
                policy_context,
                class_families,
            )
            tree = self._apply_helper_consolidation(
                tree,
                helper_mappings,
                changes,
                policy_context,
                helper_families,
            )
            tree = self._apply_nested_class_propagation(
                tree,
                class_renames,
                changes,
                policy_context,
                class_families,
            )

            result_code = tree.code
            modified = result_code != source

            if modified:
                post_payload = self._build_postcheck_payload(
                    mappings,
                    file_path,
                    confidence_threshold,
                )
                post_ok, post_errors = self._post_check_gate.validate(
                    m.Infra.Refactor.Result(
                        file_path=file_path,
                        success=True,
                        modified=True,
                        changes=changes,
                        refactored_code=result_code,
                    ),
                    post_payload,
                )
                if not post_ok:
                    return m.Infra.Refactor.Result(
                        file_path=file_path,
                        success=False,
                        modified=False,
                        error="postcheck_failed",
                        changes=post_errors,
                        refactored_code=None,
                    )

            if modified and not dry_run:
                file_path.write_text(result_code, encoding=c.Infra.Encoding.DEFAULT)

            return m.Infra.Refactor.Result(
                file_path=file_path,
                success=True,
                modified=modified,
                changes=changes,
                refactored_code=result_code,
            )
        except Exception as exc:
            return m.Infra.Refactor.Result(
                file_path=file_path,
                success=False,
                modified=False,
                error=str(exc),
                changes=[],
                refactored_code=None,
            )

    def _load_config(self) -> _RuleConfig:
        raw = self._config_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
        loaded_raw = yaml.safe_load(raw)
        try:
            loaded = TypeAdapter(dict[str, t.ContainerValue]).validate_python(
                loaded_raw
            )
        except ValidationError:
            msg = "invalid class nesting mapping config"
            raise ValueError(msg) from None

        config: _RuleConfig = {}
        confidence_threshold = loaded.get("confidence_threshold")
        if isinstance(confidence_threshold, str):
            config["confidence_threshold"] = confidence_threshold

        class_nesting_raw = loaded.get(c.Infra.ReportKeys.CLASS_NESTING)
        if isinstance(class_nesting_raw, list):
            config[c.Infra.ReportKeys.CLASS_NESTING] = self._coerce_entries(
                _mapping_list(class_nesting_raw)
            )

        helper_raw = loaded.get(c.Infra.ReportKeys.HELPER_CONSOLIDATION)
        if isinstance(helper_raw, list):
            config[c.Infra.ReportKeys.HELPER_CONSOLIDATION] = self._coerce_entries(
                _mapping_list(helper_raw)
            )

        return config

    def _confidence_threshold(self, config: _RuleConfig) -> str:
        raw = config.get("confidence_threshold", c.Infra.Severity.LOW)
        if not isinstance(raw, str):
            msg = "confidence_threshold must be a string"
            raise ValueError(msg)
        candidate = raw.strip().lower()
        if candidate in c.Infra.Refactor.CONFIDENCE_RANKS:
            return candidate
        msg = f"unsupported confidence_threshold: {raw}"
        raise ValueError(msg)

    def _confidence_allowed(self, confidence: str, threshold: str) -> bool:
        confidence_rank = c.Infra.Refactor.CONFIDENCE_RANKS.get(
            confidence.strip().lower(), 0
        )
        threshold_rank = c.Infra.Refactor.CONFIDENCE_RANKS.get(threshold, 0)
        return confidence_rank >= threshold_rank

    def _class_nesting_mappings(
        self,
        config: _RuleConfig,
        file_path: Path,
        confidence_threshold: str,
    ) -> dict[str, str]:
        mappings: dict[str, str] = {}
        for entry in self._entries_for_source_file(
            _entry_list(config.get(c.Infra.ReportKeys.CLASS_NESTING)),
            file_path,
            confidence_threshold,
        ):
            loose_name = entry.get(c.Infra.ReportKeys.LOOSE_NAME)
            target_namespace = entry.get(c.Infra.ReportKeys.TARGET_NAMESPACE)
            if isinstance(loose_name, str) and isinstance(target_namespace, str):
                mappings[loose_name] = target_namespace
        return mappings

    def _run_precheck(
        self,
        config: _RuleConfig,
        file_path: Path,
        confidence_threshold: str,
    ) -> list[str]:
        violations: list[str] = []
        entries = self._entries_for_source_file(
            _entry_list(config.get(c.Infra.ReportKeys.CLASS_NESTING)),
            file_path,
            confidence_threshold,
        )
        helper_entries = self._entries_for_source_file(
            _entry_list(config.get(c.Infra.ReportKeys.HELPER_CONSOLIDATION)),
            file_path,
            confidence_threshold,
        )
        entries.extend(helper_entries)
        for entry in entries:
            ok, violation = self._pre_check_gate.validate_entry(entry)
            if not ok and violation is not None:
                violations.append(
                    "|".join([
                        violation[c.Infra.ReportKeys.RULE_ID],
                        violation[c.Infra.ReportKeys.SOURCE_SYMBOL],
                        violation[c.Infra.ReportKeys.VIOLATION_TYPE],
                        violation[c.Infra.ReportKeys.SUGGESTED_FIX],
                    ])
                )
        return violations

    def _helper_consolidation_mappings(
        self,
        config: _RuleConfig,
        file_path: Path,
        confidence_threshold: str,
    ) -> dict[str, str]:
        mappings: dict[str, str] = {}
        for entry in self._entries_for_source_file(
            _entry_list(config.get(c.Infra.ReportKeys.HELPER_CONSOLIDATION)),
            file_path,
            confidence_threshold,
        ):
            helper_name = entry.get("helper_name")
            target_namespace = entry.get(c.Infra.ReportKeys.TARGET_NAMESPACE)
            if isinstance(helper_name, str) and isinstance(target_namespace, str):
                mappings[helper_name] = target_namespace
        return mappings

    def _class_rename_mappings(
        self,
        config: _RuleConfig,
        file_path: Path,
        confidence_threshold: str,
    ) -> dict[str, str]:
        mappings: dict[str, str] = {}
        for entry in self._entries_for_scope(
            _entry_list(config.get(c.Infra.ReportKeys.CLASS_NESTING)),
            file_path,
            confidence_threshold,
        ):
            loose_name = entry.get(c.Infra.ReportKeys.LOOSE_NAME)
            target_namespace = entry.get(c.Infra.ReportKeys.TARGET_NAMESPACE)
            target_name = entry.get("target_name")
            if not isinstance(loose_name, str):
                continue
            if not isinstance(target_namespace, str):
                continue
            if not isinstance(target_name, str):
                continue
            mappings[loose_name] = f"{target_namespace}.{target_name}"
        return mappings

    def _entries_for_source_file(
        self,
        raw_entries: list[_MappingEntry],
        file_path: Path,
        confidence_threshold: str,
    ) -> list[_MappingEntry]:
        entries = raw_entries
        if not entries:
            return []

        module_path = self._normalize_module_path(file_path)
        accepted: list[_MappingEntry] = []
        for entry in entries:
            current_file = entry.get(c.Infra.ReportKeys.CURRENT_FILE)
            if current_file is None:
                continue

            current_module = self._normalize_module_path(Path(current_file))
            if current_module != module_path and not module_path.endswith(
                f"/{current_module}"
            ):
                continue

            confidence = entry.get(c.Infra.ReportKeys.CONFIDENCE, c.Infra.Severity.LOW)
            if not self._confidence_allowed(confidence, confidence_threshold):
                continue

            accepted.append(entry)
        return accepted

    def _entries_for_scope(
        self,
        raw_entries: list[_MappingEntry],
        file_path: Path,
        confidence_threshold: str,
    ) -> list[_MappingEntry]:
        entries = raw_entries
        if not entries:
            return []

        accepted: list[_MappingEntry] = []
        for entry in entries:
            confidence = entry.get(c.Infra.ReportKeys.CONFIDENCE, c.Infra.Severity.LOW)
            if not self._confidence_allowed(confidence, confidence_threshold):
                continue

            current_file = entry.get(c.Infra.ReportKeys.CURRENT_FILE, "")
            if not current_file:
                continue

            if not self._scope_applies_to_file(entry, Path(current_file), file_path):
                continue

            accepted.append(entry)
        return accepted

    def _coerce_entries(
        self,
        entries: list[dict[str, t.ContainerValue]],
    ) -> list[_MappingEntry]:
        coerced: list[_MappingEntry] = []
        for typed in entries:
            current_file = typed.get(c.Infra.ReportKeys.CURRENT_FILE)
            if not isinstance(current_file, str):
                continue

            entry: _MappingEntry = {c.Infra.ReportKeys.CURRENT_FILE: current_file}
            loose_name = typed.get(c.Infra.ReportKeys.LOOSE_NAME)
            if isinstance(loose_name, str):
                entry[c.Infra.ReportKeys.LOOSE_NAME] = loose_name
            helper_name = typed.get("helper_name")
            if isinstance(helper_name, str):
                entry["helper_name"] = helper_name
            target_namespace = typed.get(c.Infra.ReportKeys.TARGET_NAMESPACE)
            if isinstance(target_namespace, str):
                entry[c.Infra.ReportKeys.TARGET_NAMESPACE] = target_namespace
            target_name = typed.get("target_name")
            if isinstance(target_name, str):
                entry["target_name"] = target_name
            rewrite_scope = typed.get(c.Infra.ReportKeys.REWRITE_SCOPE)
            if isinstance(rewrite_scope, str):
                entry[c.Infra.ReportKeys.REWRITE_SCOPE] = rewrite_scope
            confidence = typed.get(c.Infra.ReportKeys.CONFIDENCE)
            if isinstance(confidence, str):
                entry[c.Infra.ReportKeys.CONFIDENCE] = confidence
            coerced.append(entry)
        return coerced

    def _build_postcheck_payload(
        self,
        config: _RuleConfig,
        file_path: Path,
        confidence_threshold: str,
    ) -> _PostCheckPayload:
        payload: _PostCheckPayload = {
            c.Infra.ReportKeys.SOURCE_SYMBOL: "",
            "expected_base_chain": [],
            c.Infra.ReportKeys.POST_CHECKS: ["imports_resolve", "mro_valid"],
            "quality_gates": ["lsp_diagnostics_clean"],
        }
        class_entries = self._entries_for_source_file(
            _entry_list(config.get(c.Infra.ReportKeys.CLASS_NESTING)),
            file_path,
            confidence_threshold,
        )
        if not class_entries:
            return payload

        source_symbol = class_entries[0].get(c.Infra.ReportKeys.LOOSE_NAME, "")
        if not source_symbol:
            return payload

        payload[c.Infra.ReportKeys.SOURCE_SYMBOL] = source_symbol
        policy_doc = self._load_policy_document()
        rules = _mapping_list(policy_doc.get(c.Infra.ReportKeys.RULES))
        for rule in rules:
            if rule.get(c.Infra.ReportKeys.SOURCE_SYMBOL, "") != source_symbol:
                continue

            payload["expected_base_chain"] = _string_list(
                rule.get("expected_base_chain")
            )

            post_checks_raw = rule.get(c.Infra.ReportKeys.POST_CHECKS, [])
            post_checks: list[str] = []
            if not isinstance(post_checks_raw, list):
                continue
            checks = _mapping_list(post_checks_raw)
            for check in checks:
                check_type = check.get("type")
                if isinstance(check_type, str):
                    post_checks.append(check_type)
            if post_checks:
                payload[c.Infra.ReportKeys.POST_CHECKS] = post_checks
            break

        return payload

    def _load_policy_document(self) -> _PolicyDocument:
        try:
            loaded_raw = yaml.safe_load(
                self._policy_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            )
        except (OSError, yaml.YAMLError):
            msg = f"failed to read policy document: {self._policy_path}"
            raise ValueError(msg) from None
        try:
            loaded = TypeAdapter(dict[str, t.ContainerValue]).validate_python(
                loaded_raw
            )
        except ValidationError:
            msg = "policy document must be a mapping"
            raise ValueError(msg) from None
        if not self._policy_document_schema_valid(loaded):
            msg = "policy document failed schema validation"
            raise ValueError(msg)
        return loaded

    def _policy_document_schema_valid(self, loaded: _PolicyDocument) -> bool:
        schema_path = self._policy_path.with_name("class-policy-v2.schema.json")
        schema_raw = FlextInfraJsonService().load(schema_path)
        if schema_raw is None:
            return False
        try:
            schema = TypeAdapter(dict[str, t.ContainerValue]).validate_python(
                schema_raw
            )
        except ValidationError:
            return False

        top_required = _string_list(schema.get("required", []))
        if not self._has_required_fields(loaded, top_required):
            return False

        definitions_raw = schema.get("definitions", {})
        try:
            definitions = TypeAdapter(dict[str, t.ContainerValue]).validate_python(
                definitions_raw
            )
        except ValidationError:
            return False

        policy_entry_raw = definitions.get("policyEntry", {})
        class_rule_raw = definitions.get("classRule", {})
        try:
            policy_entry = TypeAdapter(dict[str, t.ContainerValue]).validate_python(
                policy_entry_raw
            )
            class_rule = TypeAdapter(dict[str, t.ContainerValue]).validate_python(
                class_rule_raw
            )
        except ValidationError:
            return False

        policy_entry_required = _string_list(policy_entry.get("required", []))
        class_rule_required = _string_list(class_rule.get("required", []))

        for entry in _mapping_list(loaded.get("policy_matrix")):
            if not self._has_required_fields(entry, policy_entry_required):
                return False

        for rule in _mapping_list(loaded.get(c.Infra.ReportKeys.RULES)):
            if not self._has_required_fields(rule, class_rule_required):
                return False

        return True

    def _has_required_fields(
        self, entry: t.ContainerValue, required_fields: list[str]
    ) -> bool:
        if not isinstance(entry, dict):
            return False
        return all(key in entry for key in required_fields)

    def _scope_applies_to_file(
        self,
        entry: _MappingEntry,
        current_file: Path,
        candidate_file: Path,
    ) -> bool:
        rewrite_scope = self._rewrite_scope(entry)
        if rewrite_scope == c.Infra.ReportKeys.WORKSPACE:
            return True

        current_module = self._normalize_module_path(current_file)
        candidate_module = self._normalize_module_path(candidate_file)
        if rewrite_scope == c.Infra.ReportKeys.FILE:
            return current_module == candidate_module

        current_tokens = self._project_scope_tokens(current_file)
        candidate_tokens = self._project_scope_tokens(candidate_file)
        if current_tokens and candidate_tokens:
            return bool(current_tokens & candidate_tokens)

        return current_module == candidate_module

    def _rewrite_scope(self, entry: _MappingEntry) -> str:
        raw_scope = entry.get(c.Infra.ReportKeys.REWRITE_SCOPE, c.Infra.ReportKeys.FILE)
        scope = raw_scope.strip().lower()
        if scope in {
            c.Infra.ReportKeys.FILE,
            c.Infra.Toml.PROJECT,
            c.Infra.ReportKeys.WORKSPACE,
        }:
            return scope
        msg = f"unsupported rewrite_scope: {raw_scope}"
        raise ValueError(msg)

    def _project_scope_tokens(self, path_value: Path) -> set[str]:
        normalized = path_value.as_posix().replace("\\", "/")
        parts = Path(normalized).parts
        if not parts:
            return set()

        tokens: set[str] = set()
        if c.Infra.Paths.DEFAULT_SRC_DIR in parts:
            src_index = parts.index(c.Infra.Paths.DEFAULT_SRC_DIR)
            if src_index > 0:
                tokens.add(parts[src_index - 1])
            if src_index + 1 < len(parts):
                tokens.add(parts[src_index + 1])
        return tokens

    def _normalize_module_path(self, path_value: Path) -> str:
        normalized = path_value.as_posix().replace("\\", "/")
        path = Path(normalized)
        parts = path.parts
        if c.Infra.Paths.DEFAULT_SRC_DIR in parts:
            src_index = parts.index(c.Infra.Paths.DEFAULT_SRC_DIR)
            suffix = parts[src_index + 1 :]
            if suffix:
                return Path(*suffix).as_posix()
        return path.as_posix().lstrip("./")

    def _apply_class_nesting(
        self,
        tree: cst.Module,
        mappings: dict[str, str],
        changes: list[str],
        policy_context: t.Infra.PolicyContext,
        class_families: t.Infra.ClassFamilyMap,
    ) -> cst.Module:
        transformer = FlextInfraRefactorClassNestingTransformer(
            mappings=mappings,
            policy_context=policy_context,
            class_families=class_families,
        )
        updated_tree = tree.visit(transformer)
        if updated_tree.code != tree.code:
            changes.append(
                f"Applied FlextInfraRefactorClassNestingTransformer ({len(mappings)} mappings)"
            )
        return updated_tree

    def _apply_helper_consolidation(
        self,
        tree: cst.Module,
        mappings: dict[str, str],
        changes: list[str],
        policy_context: t.Infra.PolicyContext,
        helper_families: t.Infra.ClassFamilyMap,
    ) -> cst.Module:
        transformer = HelperConsolidationTransformer(
            helper_mappings=mappings,
            policy_context=policy_context,
            helper_families=helper_families,
        )
        updated_tree = tree.visit(transformer)
        if updated_tree.code != tree.code:
            changes.append(
                f"Applied HelperConsolidationTransformer ({len(mappings)} mappings)"
            )
        return updated_tree

    def _apply_nested_class_propagation(
        self,
        tree: cst.Module,
        mappings: dict[str, str],
        changes: list[str],
        policy_context: t.Infra.PolicyContext,
        class_families: t.Infra.ClassFamilyMap,
    ) -> cst.Module:
        transformer = NestedClassPropagationTransformer(
            class_renames=mappings,
            policy_context=policy_context,
            class_families=class_families,
        )
        wrapped_tree = MetadataWrapper(tree)
        updated_tree = wrapped_tree.visit(transformer)
        if updated_tree.code != tree.code:
            changes.append(
                f"Applied NestedClassPropagationTransformer ({len(mappings)} renames)"
            )
        return updated_tree

    def _policy_context_from_document(self) -> t.Infra.PolicyContext:
        policy_doc = self._load_policy_document()
        policy_entries = FlextInfraRefactorClassNestingRuleUtilities.mapping_list(
            policy_doc.get("policy_matrix")
        )
        policy_context: dict[str, object] = {}
        for entry in policy_entries:
            family_name = entry.get("family_name")
            if not isinstance(family_name, str):
                continue
            policy_context[family_name] = entry
        return policy_context

    def _families_for_scope(
        self,
        *,
        entries: list[_MappingEntry],
        symbol_key: str,
    ) -> dict[str, str]:
        families: dict[str, str] = {}
        for entry in entries:
            symbol = entry.get(symbol_key)
            current_file = entry.get(c.Infra.ReportKeys.CURRENT_FILE)
            if not isinstance(symbol, str) or not isinstance(current_file, str):
                continue
            families[symbol] = (
                refactor_utilities.FlextInfraRefactorUtilities.module_family_from_path(
                    current_file
                )
            )
        return families


__all__ = ["ClassNestingRefactorRule"]
