"""LibCST-driven migration of module constants into MRO constants facades."""

from __future__ import annotations

from pathlib import Path

import libcst as cst

from flext_infra import c, m, u
from flext_infra.refactor.transformers.mro_private_inline import (
    FlextInfraRefactorMROPrivateInlineTransformer,
    FlextInfraRefactorMROQualifiedReferenceTransformer,
)


class FlextInfraRefactorMROMigrationTransformer:
    """Move module-level constants into the constants facade class."""

    @staticmethod
    def migrate_file(
        *,
        scan_result: m.Infra.Refactor.MROScanReport,
    ) -> tuple[str, m.Infra.Refactor.MROFileMigration, dict[str, str]]:
        """Transform a candidate file and return code plus symbol map."""
        source = Path(scan_result.file).read_text(encoding=c.Infra.Encoding.DEFAULT)
        module = u.Infra.parse_cst_from_source(source)
        if module is None:
            return (
                source,
                m.Infra.Refactor.MROFileMigration(
                    file=scan_result.file,
                    module=scan_result.module,
                    moved_symbols=(),
                    created_classes=(),
                ),
                {},
            )
        candidate_symbols = {candidate.symbol for candidate in scan_result.candidates}
        moved_statements: list[tuple[str, cst.CSTNode]] = []
        retained_module_body: list[cst.CSTNode] = []
        for stmt in module.body:
            moved = FlextInfraRefactorMROMigrationTransformer._extract_moved_statement(
                statement=stmt,
                candidate_symbols=candidate_symbols,
            )
            if moved is None:
                retained_module_body.append(stmt)
                continue
            moved_statements.append(moved)
        if len(moved_statements) == 0:
            return (
                source,
                m.Infra.Refactor.MROFileMigration(
                    file=scan_result.file,
                    module=scan_result.module,
                    moved_symbols=(),
                    created_classes=(),
                ),
                {},
            )
        moved_by_symbol = dict(moved_statements)
        ordered_symbols = [symbol for symbol, _ in moved_statements]
        transformed_body: list[cst.CSTNode] = []
        symbol_map: dict[str, str] = {}
        class_name = (
            scan_result.constants_class or c.Infra.Refactor.DEFAULT_CONSTANTS_CLASS
        )
        class_found = False
        for retained_stmt in retained_module_body:
            if (
                isinstance(retained_stmt, cst.ClassDef)
                and retained_stmt.name.value == class_name
            ):
                class_found = True
                transformed_class, class_symbol_map = (
                    FlextInfraRefactorMROMigrationTransformer._migrate_constants_class(
                        class_def=retained_stmt,
                        moved_by_symbol=moved_by_symbol,
                        ordered_symbols=ordered_symbols,
                    )
                )
                transformed_body.append(transformed_class)
                symbol_map.update(class_symbol_map)
                continue
            transformed_body.append(retained_stmt)
        created_classes: tuple[str, ...] = ()
        if not class_found:
            new_class, class_symbol_map = (
                FlextInfraRefactorMROMigrationTransformer._create_constants_class(
                    class_name=class_name,
                    moved_by_symbol=moved_by_symbol,
                    ordered_symbols=ordered_symbols,
                )
            )
            transformed_body.append(new_class)
            symbol_map.update(class_symbol_map)
            created_classes = (class_name,)
        updated_module = module.with_changes(body=tuple(transformed_body))
        replacement_values: dict[str, cst.BaseExpression] = {}
        for symbol in ordered_symbols:
            if not symbol.startswith("_"):
                continue
            value = FlextInfraRefactorMROMigrationTransformer._statement_value(
                statement=moved_by_symbol[symbol],
            )
            if value is None:
                continue
            replacement_values[symbol] = value
        inline_transformer = FlextInfraRefactorMROPrivateInlineTransformer(
            replacement_values=replacement_values,
        )
        updated_module = updated_module.visit(inline_transformer)
        facade_alias = scan_result.facade_alias or class_name
        qualified_renames: dict[str, cst.BaseExpression] = {}
        for symbol, target_path in symbol_map.items():
            if symbol.startswith("_") or "." not in target_path:
                continue
            parts = target_path.split(".")
            current: cst.BaseExpression = cst.Name(facade_alias)
            for part in parts:
                current = cst.Attribute(value=current, attr=cst.Name(part))
            qualified_renames[symbol] = current
        if len(qualified_renames) > 0:
            qualified_transformer = FlextInfraRefactorMROQualifiedReferenceTransformer(
                renames=qualified_renames,
            )
            updated_module = updated_module.visit(qualified_transformer)
        migration = m.Infra.Refactor.MROFileMigration(
            file=scan_result.file,
            module=scan_result.module,
            moved_symbols=tuple(ordered_symbols),
            created_classes=created_classes,
        )
        return (updated_module.code, migration, symbol_map)

    @staticmethod
    def _extract_moved_statement(
        *,
        statement: cst.CSTNode,
        candidate_symbols: set[str],
    ) -> tuple[str, cst.CSTNode] | None:
        if isinstance(statement, cst.ClassDef):
            symbol = statement.name.value
            if symbol in candidate_symbols:
                return (symbol, statement)
            return None
        if isinstance(statement, cst.TypeAlias):
            symbol = statement.name.value
            if symbol not in candidate_symbols:
                return None
            return (symbol, statement)
        if not isinstance(statement, cst.SimpleStatementLine):
            return None
        if len(statement.body) != 1:
            return None
        first_stmt = statement.body[0]
        if isinstance(first_stmt, cst.AnnAssign):
            if not isinstance(first_stmt.target, cst.Name):
                return None
            symbol = first_stmt.target.value
        elif isinstance(first_stmt, cst.Assign):
            if len(first_stmt.targets) != 1:
                return None
            assign_target = first_stmt.targets[0].target
            if not isinstance(assign_target, cst.Name):
                return None
            symbol = assign_target.value
        elif isinstance(first_stmt, cst.TypeAlias):
            symbol = first_stmt.name.value
        else:
            return None
        if symbol not in candidate_symbols:
            return None
        return (symbol, first_stmt)

    @staticmethod
    def _migrate_constants_class(
        *,
        class_def: cst.ClassDef,
        moved_by_symbol: dict[str, cst.CSTNode],
        ordered_symbols: list[str],
    ) -> tuple[cst.ClassDef, dict[str, str]]:
        retained_class_body: list[cst.CSTNode] = []
        alias_by_symbol: dict[str, str] = {}
        alias_replacement_values: dict[str, cst.BaseExpression] = {}
        is_types_facade = class_def.name.value.endswith("Types")
        for statement in class_def.body.body:
            alias = FlextInfraRefactorMROMigrationTransformer._extract_alias_assignment(
                statement=statement,
            )
            if alias is not None and alias[1] in moved_by_symbol:
                alias_by_symbol[alias[1]] = alias[0]
                private_value = (
                    FlextInfraRefactorMROMigrationTransformer._statement_value(
                        statement=moved_by_symbol[alias[1]],
                    )
                )
                if private_value is not None:
                    alias_replacement_values[alias[0]] = private_value
                continue
            retained_class_body.append(statement)
        symbol_map: dict[str, str] = {}
        added_targets: set[str] = set()
        moved_lines: list[cst.CSTNode] = []
        moved_core_lines: list[cst.CSTNode] = []
        for symbol in ordered_symbols:
            target = alias_by_symbol.get(
                symbol,
            ) or FlextInfraRefactorMROMigrationTransformer._default_target(
                symbol=symbol,
            )
            moved_statement = moved_by_symbol[symbol]
            use_core_namespace = is_types_facade and isinstance(
                moved_statement, cst.TypeAlias
            )
            map_target = f"Core.{target}" if use_core_namespace else target
            if map_target in added_targets:
                continue
            added_targets.add(map_target)
            symbol_map[symbol] = map_target
            replacement_value = alias_replacement_values.get(target)
            moved_node = FlextInfraRefactorMROMigrationTransformer._retarget_statement(
                statement=moved_statement,
                target_name=target,
                replacement_value=replacement_value,
            )
            if use_core_namespace:
                moved_core_lines.append(moved_node)
            else:
                moved_lines.append(moved_node)
        cleaned_body = [
            statement
            for statement in retained_class_body
            if not (
                len(moved_lines) > 0
                and isinstance(statement, cst.SimpleStatementLine)
                and (len(statement.body) == 1)
                and isinstance(statement.body[0], cst.Pass)
            )
        ]
        final_nodes: list[cst.CSTNode] = [*cleaned_body]
        if len(moved_core_lines) > 0:
            has_existing_core = any(
                isinstance(s, cst.ClassDef) and s.name.value == "Core"
                for s in final_nodes
            )
            if has_existing_core:
                merged_body, _ = (
                    FlextInfraRefactorMROMigrationTransformer._merge_core_class(
                        class_body=final_nodes,
                        moved_core_lines=moved_core_lines,
                        target_class_name="Core",
                    )
                )
                final_nodes = merged_body
            else:
                merged_body, inserted_core = (
                    FlextInfraRefactorMROMigrationTransformer._merge_core_class(
                        class_body=final_nodes,
                        moved_core_lines=moved_core_lines,
                        target_class_name="_Core",
                    )
                )
                final_nodes = merged_body
                if inserted_core and (
                    not FlextInfraRefactorMROMigrationTransformer._has_core_alias(
                        class_body=final_nodes,
                    )
                ):
                    final_nodes.append(
                        FlextInfraRefactorMROMigrationTransformer._core_alias_statement(),
                    )
        final_nodes.extend(moved_lines)
        final_body = [
            statement
            for statement in final_nodes
            if isinstance(statement, cst.BaseStatement)
        ]
        if len(final_body) == 0:
            final_body = [cst.SimpleStatementLine(body=[cst.Pass()])]
        return (
            class_def.with_changes(
                body=class_def.body.with_changes(body=tuple(final_body)),
            ),
            symbol_map,
        )

    @staticmethod
    def _create_constants_class(
        *,
        class_name: str,
        moved_by_symbol: dict[str, cst.CSTNode],
        ordered_symbols: list[str],
    ) -> tuple[cst.ClassDef, dict[str, str]]:
        class_template = cst.ClassDef(
            name=cst.Name(class_name),
            body=cst.IndentedBlock(body=()),
        )
        class_body: list[cst.BaseStatement] = []
        core_body: list[cst.BaseStatement] = []
        symbol_map: dict[str, str] = {}
        is_types_facade = class_name.endswith("Types")
        for symbol in ordered_symbols:
            target = FlextInfraRefactorMROMigrationTransformer._default_target(
                symbol=symbol,
            )
            statement = moved_by_symbol[symbol]
            use_core_namespace = is_types_facade and isinstance(
                statement, cst.TypeAlias
            )
            symbol_map[symbol] = f"Core.{target}" if use_core_namespace else target
            moved_node = FlextInfraRefactorMROMigrationTransformer._retarget_statement(
                statement=statement,
                target_name=target,
                replacement_value=None,
            )
            if isinstance(moved_node, cst.BaseStatement):
                if use_core_namespace:
                    core_body.append(moved_node)
                else:
                    class_body.append(moved_node)
        if len(core_body) > 0:
            class_body.extend(
                [
                    cst.ClassDef(
                        name=cst.Name("_Core"),
                        body=cst.IndentedBlock(body=tuple(core_body)),
                    ),
                    FlextInfraRefactorMROMigrationTransformer._core_alias_statement(),
                ],
            )
        return (
            class_template.with_changes(
                body=class_template.body.with_changes(body=tuple(class_body)),
            ),
            symbol_map,
        )

    @staticmethod
    def _extract_alias_assignment(*, statement: cst.CSTNode) -> tuple[str, str] | None:
        if not isinstance(statement, cst.SimpleStatementLine):
            return None
        if len(statement.body) != 1:
            return None
        assign = statement.body[0]
        if isinstance(assign, cst.AnnAssign):
            if not isinstance(assign.target, cst.Name):
                return None
            if not isinstance(assign.value, cst.Name):
                return None
            return (assign.target.value, assign.value.value)
        if isinstance(assign, cst.Assign):
            if len(assign.targets) != 1:
                return None
            if not isinstance(assign.targets[0].target, cst.Name):
                return None
            if not isinstance(assign.value, cst.Name):
                return None
            return (assign.targets[0].target.value, assign.value.value)
        return None

    @staticmethod
    def _default_target(*, symbol: str) -> str:
        stripped = symbol.lstrip("_")
        return stripped or symbol

    @staticmethod
    def _statement_value(*, statement: cst.CSTNode) -> cst.BaseExpression | None:
        if isinstance(statement, cst.TypeAlias):
            return statement.value
        if isinstance(statement, cst.AnnAssign):
            return statement.value
        if isinstance(statement, cst.Assign):
            return statement.value
        return None

    @staticmethod
    def _retarget_statement(
        *,
        statement: cst.CSTNode,
        target_name: str,
        replacement_value: cst.BaseExpression | None,
    ) -> cst.CSTNode:
        if isinstance(statement, cst.ClassDef):
            if statement.name.value == target_name:
                return statement
            return statement.with_changes(name=cst.Name(target_name))
        if isinstance(statement, cst.TypeAlias):
            return cst.SimpleStatementLine(
                body=[statement.with_changes(name=cst.Name(target_name))],
            )
        if isinstance(statement, cst.AnnAssign):
            if replacement_value is not None:
                return cst.SimpleStatementLine(
                    body=[
                        statement.with_changes(
                            target=cst.Name(target_name),
                            value=replacement_value,
                        ),
                    ],
                )
            return cst.SimpleStatementLine(
                body=[statement.with_changes(target=cst.Name(target_name))],
            )
        if isinstance(statement, cst.Assign):
            assign_value = replacement_value or statement.value
            return cst.SimpleStatementLine(
                body=[
                    statement.with_changes(
                        targets=(cst.AssignTarget(target=cst.Name(target_name)),),
                        value=assign_value,
                    ),
                ],
            )
        msg = "unsupported constant statement type"
        raise ValueError(msg)

    @staticmethod
    def _merge_core_class(
        *,
        class_body: list[cst.CSTNode],
        moved_core_lines: list[cst.CSTNode],
        target_class_name: str = "_Core",
    ) -> tuple[list[cst.CSTNode], bool]:
        for index, statement in enumerate(class_body):
            if not (
                isinstance(statement, cst.ClassDef)
                and statement.name.value == target_class_name
            ):
                continue
            existing_names = {
                FlextInfraRefactorMROMigrationTransformer._statement_symbol(
                    statement=item,
                )
                for item in statement.body.body
            }
            appended_lines = [
                moved
                for moved in moved_core_lines
                if FlextInfraRefactorMROMigrationTransformer._statement_symbol(
                    statement=moved,
                )
                not in existing_names
            ]
            existing_body: list[cst.BaseStatement] = [
                item
                for item in statement.body.body
                if isinstance(item, cst.BaseStatement)
            ]
            updated_core = statement.with_changes(
                body=statement.body.with_changes(
                    body=(*existing_body, *appended_lines),
                ),
            )
            merged = [*class_body]
            merged[index] = updated_core
            return (merged, False)
        merged_body = [
            *class_body,
            cst.ClassDef(
                name=cst.Name(target_class_name),
                body=cst.IndentedBlock(
                    body=tuple(
                        item
                        for item in moved_core_lines
                        if isinstance(item, cst.BaseStatement)
                    ),
                ),
                leading_lines=(cst.EmptyLine(),),
            ),
        ]
        return (merged_body, True)

    @staticmethod
    def _statement_symbol(*, statement: cst.CSTNode) -> str:
        if isinstance(statement, cst.TypeAlias):
            return statement.name.value
        if isinstance(statement, cst.SimpleStatementLine) and len(statement.body) == 1:
            base_statement = statement.body[0]
            if isinstance(base_statement, cst.TypeAlias):
                return base_statement.name.value
            if isinstance(base_statement, cst.AnnAssign) and isinstance(
                base_statement.target,
                cst.Name,
            ):
                return base_statement.target.value
            if (
                isinstance(base_statement, cst.Assign)
                and len(base_statement.targets) == 1
            ):
                assign_target = base_statement.targets[0].target
                if isinstance(assign_target, cst.Name):
                    return assign_target.value
        return ""

    @staticmethod
    def _core_alias_statement() -> cst.SimpleStatementLine:
        return cst.SimpleStatementLine(
            body=[
                cst.Assign(
                    targets=(cst.AssignTarget(target=cst.Name("Core")),),
                    value=cst.Name("_Core"),
                ),
            ],
        )

    @staticmethod
    def _has_core_alias(*, class_body: list[cst.CSTNode]) -> bool:
        for statement in class_body:
            if isinstance(statement, cst.ClassDef) and statement.name.value == "Core":
                return True
            alias = FlextInfraRefactorMROMigrationTransformer._extract_alias_assignment(
                statement=statement,
            )
            if alias == ("Core", "_Core"):
                return True
        return False


__all__ = [
    "FlextInfraRefactorMROMigrationTransformer",
]
