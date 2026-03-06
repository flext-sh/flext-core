"""Class nesting refactor rule: move loose classes under namespace classes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

import libcst as cst
import yaml
from libcst.metadata import MetadataWrapper

from flext_infra import c, m
from flext_infra.refactor.transformers.class_nesting import ClassNestingTransformer
from flext_infra.refactor.transformers.helper_consolidation import (
    HelperConsolidationTransformer,
)
from flext_infra.refactor.transformers.nested_class_propagation import (
    NestedClassPropagationTransformer,
)
from flext_infra.refactor.validation import PostCheckGate


class _MappingEntry(TypedDict, total=False):
    loose_name: str
    helper_name: str
    current_file: str
    target_namespace: str
    target_name: str
    confidence: str
    rewrite_scope: str


class _PolicyFamily(TypedDict, total=False):
    family_name: str
    allowed_operations: list[str]
    forbidden_operations: list[str]
    forbidden_targets: list[str]
    validation_requirements: dict[str, list[str]]


class _PolicyDocument(TypedDict, total=False):
    policy_matrix: list[_PolicyFamily]
    rules: list[_PolicyRule]


class _PolicyCheck(TypedDict, total=False):
    type: str


class _PolicyRule(TypedDict, total=False):
    source_symbol: str
    expected_base_chain: list[str]
    post_checks: list[_PolicyCheck]


class _RuleConfig(TypedDict, total=False):
    confidence_threshold: str
    class_nesting: list[_MappingEntry]
    helper_consolidation: list[_MappingEntry]


class _PostCheckPayload(TypedDict, total=False):
    source_symbol: str
    expected_base_chain: list[str]
    post_checks: list[str]
    quality_gates: list[str]


class _PreCheckViolation(TypedDict):
    rule_id: str
    source_symbol: str
    violation_type: str
    suggested_fix: str


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
        source_symbol = entry.get("loose_name", "")
        helper_symbol = entry.get("helper_name", "")
        symbol = source_symbol or helper_symbol
        target_namespace = entry.get("target_namespace", "")
        current_file = entry.get("current_file", "")
        if not symbol or not target_namespace or not current_file:
            return True, None

        module_family = self._module_family_from_path(current_file)
        policy = self._policy_by_family.get(module_family)
        if policy is None:
            return False, {
                "rule_id": f"precheck:{symbol}",
                "source_symbol": symbol,
                "violation_type": "unknown_module_family",
                "suggested_fix": f"declare explicit policy for {module_family}",
            }

        operation = "helper_consolidation" if helper_symbol else "class_nesting"
        allowed_operations = policy.get("allowed_operations", [])
        if operation not in allowed_operations:
            return False, {
                "rule_id": f"precheck:{symbol}",
                "source_symbol": symbol,
                "violation_type": "operation_not_allowed",
                "suggested_fix": f"allow {operation} in policy for {module_family}",
            }

        forbidden_operations = policy.get("forbidden_operations", [])
        if operation in forbidden_operations:
            return False, {
                "rule_id": f"precheck:{symbol}",
                "source_symbol": symbol,
                "violation_type": "operation_forbidden",
                "suggested_fix": f"remove {operation} from forbidden_operations for {module_family}",
            }

        forbidden_targets = policy.get("forbidden_targets", [])
        if any(
            self._target_matches(target_namespace, pattern)
            for pattern in forbidden_targets
        ):
            return False, {
                "rule_id": f"precheck:{symbol}",
                "source_symbol": symbol,
                "violation_type": "forbidden_target",
                "suggested_fix": f"choose allowed target for family {module_family}",
            }
        return True, None

    def _load_policy(self) -> dict[str, _PolicyFamily]:
        try:
            loaded = cast(
                "_PolicyDocument",
                yaml.safe_load(
                    self._policy_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
                ),
            )
        except (OSError, yaml.YAMLError):
            return {}
        if not self._schema_valid(loaded):
            return {}
        policy_matrix = loaded.get("policy_matrix", [])

        by_family: dict[str, _PolicyFamily] = {}
        for raw in policy_matrix:
            family_name = raw.get("family_name")
            if family_name is None:
                continue
            forbidden_targets = list(raw.get("forbidden_targets", []))

            allowed_operations_raw = raw.get("allowed_operations", [])
            allowed_operations = list(allowed_operations_raw)

            forbidden_operations_raw = raw.get("forbidden_operations", [])
            forbidden_operations = list(forbidden_operations_raw)

            validation_requirements = raw.get("validation_requirements", {})
            by_family[family_name] = {
                "family_name": family_name,
                "allowed_operations": allowed_operations,
                "forbidden_operations": forbidden_operations,
                "forbidden_targets": forbidden_targets,
                "validation_requirements": validation_requirements,
            }
        return by_family

    def _schema_valid(self, loaded: _PolicyDocument) -> bool:
        try:
            schema_raw = self._schema_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            schema = json.loads(schema_raw)
        except (OSError, json.JSONDecodeError):
            return False

        top_required = schema.get("required", [])
        if not self._has_required_fields(loaded, top_required):
            return False

        definitions = schema.get("definitions", {})
        policy_entry_required = definitions.get("policyEntry", {}).get("required", [])
        class_rule_required = definitions.get("classRule", {}).get("required", [])

        policy_matrix = loaded.get("policy_matrix", [])
        for entry in policy_matrix:
            if not self._has_required_fields(entry, policy_entry_required):
                return False

        rules = loaded.get("rules", [])
        for rule in rules:
            if not self._has_required_fields(rule, class_rule_required):
                return False

        return True

    def _has_required_fields(self, entry: object, required_fields: object) -> bool:
        if not isinstance(entry, dict):
            return False
        if not isinstance(required_fields, list):
            return True

        required_items = cast("list[object]", required_fields)
        keys = [candidate for candidate in required_items if isinstance(candidate, str)]

        return all(key in entry for key in keys)

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
            tree = self._apply_class_nesting(tree, class_mappings, changes)
            tree = self._apply_helper_consolidation(tree, helper_mappings, changes)
            tree = self._apply_nested_class_propagation(tree, class_renames, changes)

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
        loaded = yaml.safe_load(raw)
        if not isinstance(loaded, dict):
            return {}

        typed = cast("dict[str, str | list[dict[str, str]]]", loaded)
        config: _RuleConfig = {}
        confidence_threshold = typed.get("confidence_threshold")
        if isinstance(confidence_threshold, str):
            config["confidence_threshold"] = confidence_threshold

        class_nesting_raw = typed.get("class_nesting")
        if isinstance(class_nesting_raw, list):
            config["class_nesting"] = self._coerce_entries(class_nesting_raw)

        helper_raw = typed.get("helper_consolidation")
        if isinstance(helper_raw, list):
            config["helper_consolidation"] = self._coerce_entries(helper_raw)

        return config

    def _confidence_threshold(self, config: _RuleConfig) -> str:
        raw = config.get("confidence_threshold", "low")
        candidate = raw.strip().lower()
        if candidate in c.Infra.Refactor.CONFIDENCE_RANKS:
            return candidate
        return "low"

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
            config.get("class_nesting", []),
            file_path,
            confidence_threshold,
        ):
            loose_name = entry.get("loose_name")
            target_namespace = entry.get("target_namespace")
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
            config.get("class_nesting", []),
            file_path,
            confidence_threshold,
        )
        helper_entries = self._entries_for_source_file(
            config.get("helper_consolidation", []),
            file_path,
            confidence_threshold,
        )
        entries.extend(helper_entries)
        for entry in entries:
            ok, violation = self._pre_check_gate.validate_entry(entry)
            if not ok and violation is not None:
                violations.append(
                    "|".join([
                        violation["rule_id"],
                        violation["source_symbol"],
                        violation["violation_type"],
                        violation["suggested_fix"],
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
            config.get("helper_consolidation", []),
            file_path,
            confidence_threshold,
        ):
            helper_name = entry.get("helper_name")
            target_namespace = entry.get("target_namespace")
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
            config.get("class_nesting", []),
            file_path,
            confidence_threshold,
        ):
            loose_name = entry.get("loose_name")
            target_namespace = entry.get("target_namespace")
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
            current_file = entry.get("current_file")
            if current_file is None:
                continue

            current_module = self._normalize_module_path(Path(current_file))
            if current_module != module_path:
                continue

            confidence = entry.get("confidence", "low")
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
            confidence = entry.get("confidence", "low")
            if not self._confidence_allowed(confidence, confidence_threshold):
                continue

            current_file = entry.get("current_file", "")
            if not current_file:
                continue

            if not self._scope_applies_to_file(entry, Path(current_file), file_path):
                continue

            accepted.append(entry)
        return accepted

    def _coerce_entries(self, entries: list[dict[str, str]]) -> list[_MappingEntry]:
        coerced: list[_MappingEntry] = []
        for typed in entries:
            current_file = typed.get("current_file")
            if current_file is None:
                continue

            entry: _MappingEntry = {"current_file": current_file}
            loose_name = typed.get("loose_name")
            if loose_name is not None:
                entry["loose_name"] = loose_name
            helper_name = typed.get("helper_name")
            if helper_name is not None:
                entry["helper_name"] = helper_name
            target_namespace = typed.get("target_namespace")
            if target_namespace is not None:
                entry["target_namespace"] = target_namespace
            target_name = typed.get("target_name")
            if target_name is not None:
                entry["target_name"] = target_name
            rewrite_scope = typed.get("rewrite_scope")
            if rewrite_scope is not None:
                entry["rewrite_scope"] = rewrite_scope
            confidence = typed.get("confidence")
            if confidence is not None:
                entry["confidence"] = confidence
            coerced.append(entry)
        return coerced

    def _build_postcheck_payload(
        self,
        config: _RuleConfig,
        file_path: Path,
        confidence_threshold: str,
    ) -> _PostCheckPayload:
        payload: _PostCheckPayload = {
            "source_symbol": "",
            "expected_base_chain": [],
            "post_checks": ["imports_resolve", "mro_valid"],
            "quality_gates": ["lsp_diagnostics_clean"],
        }
        class_entries = self._entries_for_source_file(
            config.get("class_nesting", []),
            file_path,
            confidence_threshold,
        )
        if not class_entries:
            return payload

        source_symbol = class_entries[0].get("loose_name", "")
        if not source_symbol:
            return payload

        payload["source_symbol"] = source_symbol
        policy_doc = self._load_policy_document()
        rules = policy_doc.get("rules", [])
        for rule in rules:
            if rule.get("source_symbol", "") != source_symbol:
                continue

            expected_chain = rule.get("expected_base_chain", [])
            payload["expected_base_chain"] = list(expected_chain)

            post_checks_raw = rule.get("post_checks", [])
            post_checks: list[str] = []
            for check in post_checks_raw:
                check_type = check.get("type")
                if check_type is not None:
                    post_checks.append(check_type)
            if post_checks:
                payload["post_checks"] = post_checks
            break

        return payload

    def _load_policy_document(self) -> _PolicyDocument:
        try:
            loaded = cast(
                "_PolicyDocument",
                yaml.safe_load(
                    self._policy_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
                ),
            )
        except (OSError, yaml.YAMLError):
            return {}
        if not self._policy_document_schema_valid(loaded):
            return {}
        return loaded

    def _policy_document_schema_valid(self, loaded: _PolicyDocument) -> bool:
        schema_path = self._policy_path.with_name("class-policy-v2.schema.json")
        try:
            schema = json.loads(
                schema_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            )
        except (OSError, json.JSONDecodeError):
            return False

        top_required = schema.get("required", [])
        if not self._has_required_fields(loaded, top_required):
            return False

        definitions = schema.get("definitions", {})
        policy_entry_required = definitions.get("policyEntry", {}).get("required", [])
        class_rule_required = definitions.get("classRule", {}).get("required", [])

        for entry in loaded.get("policy_matrix", []):
            if not self._has_required_fields(entry, policy_entry_required):
                return False

        for rule in loaded.get("rules", []):
            if not self._has_required_fields(rule, class_rule_required):
                return False

        return True

    def _has_required_fields(self, entry: object, required_fields: object) -> bool:
        if not isinstance(entry, dict):
            return False
        if not isinstance(required_fields, list):
            return True

        required_items = cast("list[object]", required_fields)
        keys = [candidate for candidate in required_items if isinstance(candidate, str)]

        return all(key in entry for key in keys)

    def _scope_applies_to_file(
        self,
        entry: _MappingEntry,
        current_file: Path,
        candidate_file: Path,
    ) -> bool:
        rewrite_scope = self._rewrite_scope(entry)
        if rewrite_scope == "workspace":
            return True

        current_module = self._normalize_module_path(current_file)
        candidate_module = self._normalize_module_path(candidate_file)
        if rewrite_scope == "file":
            return current_module == candidate_module

        current_tokens = self._project_scope_tokens(current_file)
        candidate_tokens = self._project_scope_tokens(candidate_file)
        if current_tokens and candidate_tokens:
            return bool(current_tokens & candidate_tokens)

        return current_module == candidate_module

    def _rewrite_scope(self, entry: _MappingEntry) -> str:
        raw_scope = entry.get("rewrite_scope", "file")
        scope = raw_scope.strip().lower()
        if scope in {"file", "project", "workspace"}:
            return scope
        return "file"

    def _project_scope_tokens(self, path_value: Path) -> set[str]:
        normalized = path_value.as_posix().replace("\\", "/")
        parts = Path(normalized).parts
        if not parts:
            return set()

        tokens: set[str] = set()
        if "src" in parts:
            src_index = parts.index("src")
            if src_index > 0:
                tokens.add(parts[src_index - 1])
            if src_index + 1 < len(parts):
                tokens.add(parts[src_index + 1])
        return tokens

    def _normalize_module_path(self, path_value: Path) -> str:
        normalized = path_value.as_posix().replace("\\", "/")
        path = Path(normalized)
        parts = path.parts
        if "src" in parts:
            src_index = parts.index("src")
            suffix = parts[src_index + 1 :]
            if suffix:
                return Path(*suffix).as_posix()
        return path.as_posix().lstrip("./")

    def _apply_class_nesting(
        self,
        tree: cst.Module,
        mappings: dict[str, str],
        changes: list[str],
    ) -> cst.Module:
        transformer = ClassNestingTransformer(mappings=mappings)
        updated_tree = tree.visit(transformer)
        if updated_tree.code != tree.code:
            changes.append(
                f"Applied ClassNestingTransformer ({len(mappings)} mappings)"
            )
        return updated_tree

    def _apply_helper_consolidation(
        self,
        tree: cst.Module,
        mappings: dict[str, str],
        changes: list[str],
    ) -> cst.Module:
        transformer = HelperConsolidationTransformer(helper_mappings=mappings)
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
    ) -> cst.Module:
        transformer = NestedClassPropagationTransformer(class_renames=mappings)
        wrapped_tree = MetadataWrapper(tree)
        updated_tree = wrapped_tree.visit(transformer)
        if updated_tree.code != tree.code:
            changes.append(
                f"Applied NestedClassPropagationTransformer ({len(mappings)} renames)"
            )
        return updated_tree


__all__ = ["ClassNestingRefactorRule"]
