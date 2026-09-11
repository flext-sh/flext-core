"""Rope-native structural and static-analysis fact boundary (no stdlib AST).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from rope.base import codeanalyze, simplify, worder

from flext_infra._utilities.rope_core import FlextInfraUtilitiesRopeCore
from flext_infra._utilities.rope_runtime import FlextInfraUtilitiesRopeRuntime
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.protocols import p
from flext_infra.typings import t


class FlextInfraUtilitiesRopeStructure:
    """Expose one Rope-owned fact pass to every structural detector."""

    @staticmethod
    def logical_statements(source: str) -> t.SequenceOf[m.Infra.LogicalStatement]:
        """Return Rope logical regions with scope and TYPE_CHECKING context."""
        if not source:
            return ()
        lines = codeanalyze.SourceLinesAdapter(source)
        finder = codeanalyze.LogicalLineFinder(lines)
        statements: t.MutableSequenceOf[m.Infra.LogicalStatement] = []
        enclosers: t.MutableSequenceOf[tuple[int, c.Infra.RopeScopeKind, str]] = []
        type_checking_guards: t.MutableSequenceOf[int] = []
        for start, end in finder.generate_regions():
            text = "".join(lines.get_line(n) for n in range(start, end + 1))
            indent = len(text) - len(text.lstrip())
            FlextInfraUtilitiesRopeStructure._pop_exited_enclosers(enclosers, indent)
            while type_checking_guards and indent <= type_checking_guards[-1]:
                type_checking_guards.pop()
            kind, name = (
                (enclosers[-1][1], enclosers[-1][2])
                if enclosers
                else (c.Infra.RopeScopeKind.MODULE, "")
            )
            category = FlextInfraUtilitiesRopeStructure._categorize(text)
            statements.append(
                m.Infra.LogicalStatement(
                    line=start,
                    end_line=end,
                    indent=indent,
                    category=category,
                    enclosing_kind=kind,
                    enclosing_name=name,
                    type_checking_guarded=bool(type_checking_guards),
                    text=text,
                )
            )
            FlextInfraUtilitiesRopeStructure._push_encloser(
                enclosers=enclosers, category=category, indent=indent, text=text
            )
            # flext-j47u (codex): all detectors consume this single guard fact.
            if (
                category == c.Infra.StatementCategory.IF_GUARD
                and FlextInfraUtilitiesRopeStructure._is_type_checking_guard(text)
            ):
                type_checking_guards.append(indent)
        return tuple(statements)

    @classmethod
    def detect_static_rules(
        cls, ctx: m.Infra.DetectorContext, rules: t.SequenceOf[m.Infra.StaticRuleSpec]
    ) -> t.SequenceOf[m.Infra.PatternSmellViolation]:
        """Resolve one detector context and evaluate the configured Rope policy."""
        resource = FlextInfraUtilitiesRopeCore.fetch_python_resource(
            ctx.rope_project, ctx.file_path
        )
        if resource is None:
            return ()
        try:
            pymodule = FlextInfraUtilitiesRopeCore.get_pymodule(
                ctx.rope_project, resource
            )
            module_imports = FlextInfraUtilitiesRopeRuntime.module_imports_for_pymodule(
                ctx.rope_project, pymodule
            )
        except (*FlextInfraUtilitiesRopeRuntime.rope_syntax_errors(),) as exc:
            if ctx.parse_failures is None:
                raise
            ctx.parse_failures.append(
                m.Infra.ParseFailureViolation(
                    file=str(ctx.file_path),
                    stage="static_rules",
                    error_type=type(exc).__name__,
                    detail=str(exc),
                )
            )
            return ()
        display_path = (
            ctx.file_path.relative_to(ctx.project_root)
            if ctx.project_root is not None
            and ctx.file_path.is_relative_to(ctx.project_root)
            else ctx.file_path
        )
        return cls.evaluate_static_rules(
            source=resource.read(),
            module_imports=module_imports,
            rules=rules,
            file_path=str(display_path),
            project_name=ctx.project_name,
        )

    @classmethod
    def evaluate_static_rules(
        cls,
        *,
        source: str,
        module_imports: t.Infra.RopeModuleImports,
        rules: t.SequenceOf[m.Infra.StaticRuleSpec],
        file_path: str,
        project_name: str,
    ) -> t.SequenceOf[m.Infra.PatternSmellViolation]:
        """Evaluate config rules in one Rope logical-statement pass."""
        if not source:
            return ()
        lines = codeanalyze.SourceLinesAdapter(source)
        regions = cls._ignored_regions(source, lines)
        facts = cls._import_facts(module_imports)
        word_finder = worder.Worder(source, True)
        violations: tuple[m.Infra.PatternSmellViolation, ...] = ()
        for statement in cls.logical_statements(source):
            for rule in rules:
                if isinstance(rule, m.Infra.StaticCommentRule) or not cls._rule_matches(
                    rule=rule,
                    statement=statement,
                    facts=facts,
                    regions=regions,
                    source=source,
                    lines=lines,
                    word_finder=word_finder,
                    project_name=project_name,
                ):
                    continue
                violation = m.Infra.PatternSmellViolation(
                    file=file_path,
                    line=statement.line,
                    kind=rule.kind,
                    detail=rule.detail,
                )
                if violation not in violations:
                    violations = (*violations, violation)
        for region in regions:
            if not region.is_comment:
                continue
            compact = "".join(region.text.casefold().split())
            for rule in rules:
                if (
                    isinstance(rule, m.Infra.StaticCommentRule)
                    and "".join(rule.marker.casefold().split()) in compact
                ):
                    violation = m.Infra.PatternSmellViolation(
                        file=file_path,
                        line=region.line,
                        kind=rule.kind,
                        detail=rule.detail,
                    )
                    if violation not in violations:
                        violations = (*violations, violation)
        return violations

    @classmethod
    def _rule_matches(
        cls,
        *,
        rule: m.Infra.StaticRuleSpec,
        statement: m.Infra.LogicalStatement,
        facts: t.SequenceOf[m.Infra.ImportFact],
        regions: t.SequenceOf[m.Infra.IgnoredRegion],
        source: str,
        lines: codeanalyze.SourceLinesAdapter,
        word_finder: p.Infra.RopeWorder,
        project_name: str,
    ) -> bool:
        """Return whether one closed operator matches one Rope region."""
        statement_facts = tuple(
            fact for fact in facts if statement.line <= fact.line <= statement.end_line
        )
        if isinstance(rule, m.Infra.StaticImportModuleRule):
            return rule.owner_project != project_name and any(
                fact.module == rule.module or fact.module.startswith(f"{rule.module}.")
                for fact in statement_facts
            )
        if isinstance(rule, m.Infra.StaticImportMemberRule):
            return any(
                fact.is_from_import
                and fact.module == rule.module
                and fact.member == rule.member
                for fact in statement_facts
            )
        if isinstance(rule, m.Infra.StaticAttributeRule):
            return any(
                cls._primary_offsets(
                    source,
                    lines,
                    statement,
                    f"{fact.local_name}.{rule.member}",
                    regions,
                    word_finder,
                )
                for fact in facts
                if not fact.is_from_import
                and (
                    fact.module == rule.module
                    or fact.module.startswith(f"{rule.module}.")
                )
            )
        if isinstance(rule, (m.Infra.StaticCallRule, m.Infra.StaticCallKeywordRule)):
            offsets = cls._primary_offsets(
                source, lines, statement, rule.name, regions, word_finder, called=True
            )
            return bool(offsets) and (
                isinstance(rule, m.Infra.StaticCallRule)
                or any(
                    not cls._call_has_keyword(source, offset, rule.keyword, word_finder)
                    for offset in offsets
                )
            )
        if isinstance(rule, m.Infra.StaticAnnotationRule):
            return (
                statement.category == c.Infra.StatementCategory.ANN_ASSIGN
                and cls.annotation_contains(statement, rule.name)
            )
        if isinstance(rule, m.Infra.StaticBareExceptRule):
            statement_text: str = statement.text
            return statement_text.strip().partition(":")[0].strip() == "except"
        return isinstance(rule, m.Infra.StaticAnnotatedStringRule) and (
            statement.category == c.Infra.StatementCategory.ANN_ASSIGN
            and cls.target_name(statement) == rule.name
            and cls._string_assignment(statement, lines, regions)
        )

    @staticmethod
    def _primary_offsets(
        source: str,
        lines: codeanalyze.SourceLinesAdapter,
        statement: m.Infra.LogicalStatement,
        primary: str,
        regions: t.SequenceOf[m.Infra.IgnoredRegion],
        word_finder: p.Infra.RopeWorder,
        *,
        called: bool = False,
    ) -> tuple[int, ...]:
        """Return Rope-verified primary offsets inside one logical region."""
        start = lines.get_line_start(statement.line)
        end = lines.get_line_end(statement.end_line)
        offsets: tuple[int, ...] = ()
        cursor = start
        while (candidate := source.find(primary, cursor, end)) >= 0:
            cursor = candidate + len(primary)
            if any(
                region.start_offset <= candidate < region.end_offset
                for region in regions
            ):
                continue
            anchor = candidate + primary.rfind(".") + 1
            if word_finder.get_primary_at(anchor) != primary or (
                called and not word_finder.is_a_function_being_called(anchor)
            ):
                continue
            offsets = (*offsets, anchor)
        return offsets

    @staticmethod
    def _call_has_keyword(
        source: str, call_offset: int, keyword: str, word_finder: p.Infra.RopeWorder
    ) -> bool:
        """Return whether Rope recognizes a required call keyword."""
        start, end = word_finder.get_word_parens_range(call_offset)
        cursor = start
        while (candidate := source.find(keyword, cursor, end)) >= 0:
            cursor = candidate + len(keyword)
            if word_finder.is_function_keyword_parameter(candidate):
                return True
        return False

    @staticmethod
    def _import_facts(
        module_imports: t.Infra.RopeModuleImports,
    ) -> t.SequenceOf[m.Infra.ImportFact]:
        """Validate Rope NormalImport/FromImport objects into immutable facts."""
        facts: tuple[m.Infra.ImportFact, ...] = ()
        for statement in tuple(module_imports.imports):
            line = statement.start_line if statement.start_line > 0 else 1
            info = statement.import_info
            if FlextInfraUtilitiesRopeRuntime.is_normal_import(info):
                for module, alias in info.names_and_aliases:
                    facts = (
                        *facts,
                        m.Infra.ImportFact(
                            line=line,
                            module=module,
                            local_name=alias or module.partition(".")[0],
                            is_from_import=False,
                        ),
                    )
            elif FlextInfraUtilitiesRopeRuntime.is_from_import(info):
                for member, alias in info.names_and_aliases:
                    facts = (
                        *facts,
                        m.Infra.ImportFact(
                            line=line,
                            module=info.module_name,
                            member=member,
                            local_name=alias or member,
                            is_from_import=True,
                        ),
                    )
        return facts

    @staticmethod
    def _ignored_regions(
        source: str, lines: codeanalyze.SourceLinesAdapter
    ) -> t.SequenceOf[m.Infra.IgnoredRegion]:
        """Validate Rope string/comment regions into immutable facts."""
        regions: tuple[m.Infra.IgnoredRegion, ...] = ()
        for start, end, _metadata in simplify.ignored_regions(source):
            text = source[start:end]
            if text:
                regions = (
                    *regions,
                    m.Infra.IgnoredRegion(
                        line=lines.get_line_number(start),
                        start_offset=start,
                        end_offset=end,
                        text=text,
                        is_comment=text.startswith("#"),
                    ),
                )
        return regions

    @staticmethod
    def _string_assignment(
        statement: m.Infra.LogicalStatement,
        lines: codeanalyze.SourceLinesAdapter,
        regions: t.SequenceOf[m.Infra.IgnoredRegion],
    ) -> bool:
        """Return whether an annotated assignment starts with a Rope string."""
        stripped = statement.text.strip()
        head = FlextInfraUtilitiesRopeStructure._assignment_head(stripped)
        value = stripped[len(head) + 1 :].lstrip() if head is not None else ""
        relative = statement.text.find(value) if value else -1
        offset = lines.get_line_start(statement.line) + relative
        return relative >= 0 and any(
            not region.is_comment and region.start_offset == offset
            for region in regions
        )

    @staticmethod
    def _is_type_checking_guard(text: str) -> bool:
        """Return whether one logical-line region opens a TYPE_CHECKING branch."""
        normalized = " ".join(text.split())
        return normalized in {"if TYPE_CHECKING:", "if TYPE_CHECKING is True:"}

    @staticmethod
    def _pop_exited_enclosers(
        enclosers: t.MutableSequenceOf[tuple[int, c.Infra.RopeScopeKind, str]],
        indent: int,
    ) -> None:
        """Drop enclosers whose body the current indentation has left."""
        while enclosers and indent <= enclosers[-1][0]:
            enclosers.pop()

    @staticmethod
    def _push_encloser(
        *,
        enclosers: t.MutableSequenceOf[tuple[int, c.Infra.RopeScopeKind, str]],
        category: c.Infra.StatementCategory,
        indent: int,
        text: str,
    ) -> None:
        """Record a ``class``/``def`` header as an enclosing scope for its body."""
        if category == c.Infra.StatementCategory.CLASS_DEF:
            enclosers.append((
                indent,
                c.Infra.RopeScopeKind.CLASS,
                FlextInfraUtilitiesRopeStructure._header_name(text),
            ))
        elif category == c.Infra.StatementCategory.FUNC_DEF:
            enclosers.append((
                indent,
                c.Infra.RopeScopeKind.FUNCTION,
                FlextInfraUtilitiesRopeStructure._header_name(text),
            ))

    @staticmethod
    def header_name(statement: m.Infra.LogicalStatement) -> str:
        """Return the declared name from a ``class``/``def`` statement header."""
        return FlextInfraUtilitiesRopeStructure._header_name(statement.text)

    @staticmethod
    def _header_name(text: str) -> str:
        """Return the declared name from a ``class``/``def`` header line."""
        stripped = text.strip().removeprefix("async ").strip()
        body = stripped.split(maxsplit=1)[1] if " " in stripped else ""
        for separator in ("(", ":", "["):
            body = body.split(separator, maxsplit=1)[0]
        return body.strip()

    @staticmethod
    def _categorize(text: str) -> c.Infra.StatementCategory:
        """Classify one statement by its leading token (lexical, no AST)."""
        stripped = text.strip()
        first = stripped.split(maxsplit=1)[0] if stripped else ""
        keyword_map = {
            "import": c.Infra.StatementCategory.IMPORT,
            "from": c.Infra.StatementCategory.FROM_IMPORT,
            "type": c.Infra.StatementCategory.TYPE_ALIAS,
            "class": c.Infra.StatementCategory.CLASS_DEF,
            "def": c.Infra.StatementCategory.FUNC_DEF,
            "async": c.Infra.StatementCategory.FUNC_DEF,
            "if": c.Infra.StatementCategory.IF_GUARD,
        }
        if first in keyword_map:
            return keyword_map[first]
        return FlextInfraUtilitiesRopeStructure._categorize_expression(stripped)

    @staticmethod
    def _categorize_expression(stripped: str) -> c.Infra.StatementCategory:
        """Classify a non-keyword statement."""
        head = FlextInfraUtilitiesRopeStructure._assignment_head(stripped)
        if head is not None:
            return (
                c.Infra.StatementCategory.ANN_ASSIGN
                if ":" in head
                else c.Infra.StatementCategory.ASSIGN
            )
        return (
            c.Infra.StatementCategory.CALL
            if "(" in stripped
            else c.Infra.StatementCategory.OTHER
        )

    @staticmethod
    def _assignment_head(stripped: str) -> str | None:
        """Return the target side of a top-level assignment."""
        depth = 0
        quote = ""
        for index, char in enumerate(stripped):
            if quote:
                if char == quote:
                    quote = ""
            elif char in {"'", '"'}:
                quote = char
            elif char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "=" and depth == 0:
                following = stripped[index + 1] if index + 1 < len(stripped) else ""
                previous = stripped[index - 1] if index else ""
                if following != "=" and previous not in {"=", "!", "<", ">"}:
                    return stripped[:index]
        return None

    @staticmethod
    def target_name(statement: m.Infra.LogicalStatement) -> str:
        """Return a simple assignment/annotation target name, or empty."""
        head = FlextInfraUtilitiesRopeStructure._assignment_head(statement.text.strip())
        source = head if head is not None else statement.text.strip()
        name = source.split(":", maxsplit=1)[0].strip()
        return name if name.isidentifier() else ""

    @staticmethod
    def annotation_contains(statement: m.Infra.LogicalStatement, name: str) -> bool:
        """Return whether an annotation contains one identifier token."""
        annotation = FlextInfraUtilitiesRopeStructure._annotation_source(statement.text)
        return name in FlextInfraUtilitiesRopeStructure._identifiers(annotation)

    @staticmethod
    def _annotation_source(text: str) -> str:
        """Return the annotation slice of an annotated statement."""
        stripped = text.strip()
        head = FlextInfraUtilitiesRopeStructure._assignment_head(stripped)
        left = head if head is not None else stripped
        _target, separator, annotation = left.partition(":")
        return annotation.strip() if separator else ""

    @staticmethod
    def call_callee_name(statement: m.Infra.LogicalStatement) -> str:
        """Return the trailing identifier of an assignment call, or empty."""
        head = FlextInfraUtilitiesRopeStructure._assignment_head(statement.text.strip())
        if head is None:
            return ""
        value = statement.text.strip()[len(head) + 1 :].strip()
        open_paren = value.find("(")
        return (
            value[:open_paren].strip().rsplit(".", maxsplit=1)[-1]
            if open_paren > 0
            else ""
        )

    @staticmethod
    def class_base_names(statement: m.Infra.LogicalStatement) -> t.Infra.StrSet:
        """Return terminal base-class names from a class header."""
        stripped = statement.text.strip()
        open_paren = stripped.find("(")
        close_paren = stripped.rfind(")")
        if open_paren < 0 or close_paren <= open_paren:
            return set()
        return {
            terminal
            for part in stripped[open_paren + 1 : close_paren].split(",")
            if (item := part.strip())
            if (terminal := item.split("[", maxsplit=1)[0].strip().rsplit(".", 1)[-1])
        }

    @staticmethod
    def _identifiers(source: str) -> t.Infra.StrSet:
        """Return identifier tokens from a source slice."""
        identifiers: t.Infra.StrSet = set()
        token = ""
        for char in source:
            if char.isalnum() or char == "_":
                token += char
            elif token:
                identifiers.add(token)
                token = ""
        if token:
            identifiers.add(token)
        return identifiers


__all__: list[str] = ["FlextInfraUtilitiesRopeStructure"]
