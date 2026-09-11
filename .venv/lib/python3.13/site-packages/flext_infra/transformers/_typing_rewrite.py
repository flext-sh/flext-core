"""Recursive built-in container-annotation rewrite helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraRefactorTypingUnifierRewriteMixin:
    """Rewrite built-in container/tuple annotations to canonical ``t.*`` aliases.

    Composed into FlextInfraRefactorTypingUnifier via inheritance; the facade's
    ``_canonicalize_annotation_builtins`` resolves ``_rewrite_annotation_text``
    through FLEXT. Pure syntactic rewriting over text fragments (no facade state).
    """

    _CONTAINER_REWRITES: ClassVar[t.StrPairTuple] = (
        ("dict[", "t.MutableMappingKV"),
        ("Dict[", "t.MutableMappingKV"),
        ("list[", "t.MutableSequenceOf"),
        ("List[", "t.MutableSequenceOf"),
    )
    _VARIADIC_TUPLE_PARTS: ClassVar[int] = 2
    _FIXED_TUPLE_ALIASES: ClassVar[t.MappingKV[int, str]] = {
        2: "Pair",
        3: "Triple",
        4: "Quad",
    }

    def _rewrite_annotation_text(self, text: str) -> tuple[str, list[str]]:
        """Rewrite built-in container annotations found within ``text``."""
        result: list[str] = []
        changes: list[str] = []
        index = 0
        while index < len(text):
            container = self._match_container_prefix(text, index)
            if container is None:
                result.append(text[index])
                index += 1
                continue
            prefix, alias_name = container
            content, end_index = self._extract_square_bracket_content(
                text, index + len(prefix) - 1
            )
            rewritten_content, nested_changes = self._rewrite_type_expression(content)
            if prefix.lower().startswith("tuple["):
                replacement = self._rewrite_tuple_annotation(
                    original_prefix=prefix, rewritten_content=rewritten_content
                )
            else:
                replacement = f"{alias_name}[{rewritten_content}]"
            original = text[index:end_index]
            result.append(replacement)
            changes.extend(nested_changes)
            if original != replacement:
                changes.append(
                    f"Canonicalized built-in annotation {original} -> {replacement}"
                )
            index = end_index
        return "".join(result), changes

    def _rewrite_type_expression(self, text: str) -> tuple[str, list[str]]:
        """Rewrite one nested type-expression fragment recursively."""
        result: list[str] = []
        changes: list[str] = []
        index = 0
        while index < len(text):
            container = self._match_container_prefix(text, index)
            if container is not None:
                prefix, alias_name = container
                content, end_index = self._extract_square_bracket_content(
                    text, index + len(prefix) - 1
                )
                rewritten_content, nested_changes = self._rewrite_type_expression(
                    content
                )
                if prefix.lower().startswith("tuple["):
                    replacement = self._rewrite_tuple_annotation(
                        original_prefix=prefix, rewritten_content=rewritten_content
                    )
                else:
                    replacement = f"{alias_name}[{rewritten_content}]"
                original = text[index:end_index]
                result.append(replacement)
                changes.extend(nested_changes)
                if original != replacement:
                    changes.append(
                        f"Canonicalized built-in annotation {original} -> {replacement}"
                    )
                index = end_index
                continue
            token_rewrite = self._match_simple_type_alias(text, index)
            if token_rewrite is not None:
                original, replacement, end_index = token_rewrite
                result.append(replacement)
                if original != replacement:
                    changes.append(
                        f"Canonicalized built-in annotation {original} -> {replacement}"
                    )
                index = end_index
                continue
            result.append(text[index])
            index += 1
        return "".join(result), changes

    @staticmethod
    def _match_container_prefix(text: str, index: int) -> t.StrPair | None:
        """Return the matching built-in container prefix at ``index``, if any."""
        cls = FlextInfraRefactorTypingUnifierRewriteMixin
        for prefix, alias_name in cls._CONTAINER_REWRITES:
            if cls._matches_type_token(text, index, prefix):
                return prefix, alias_name
        if cls._matches_type_token(text, index, "tuple["):
            return "tuple[", ""
        if cls._matches_type_token(text, index, "Tuple["):
            return "Tuple[", ""
        return None

    @staticmethod
    def _matches_type_token(text: str, index: int, token: str) -> bool:
        """Return whether ``token`` starts at ``index`` with identifier boundaries."""
        if not text.startswith(token, index):
            return False
        before = text[index - 1] if index > 0 else ""
        return not before or not (before.isalnum() or before == "_")

    @staticmethod
    def _match_simple_type_alias(text: str, index: int) -> tuple[str, str, int] | None:
        """Return a leaf-type rewrite for ``Any``/``typing.Any``/``object``."""
        for token in ("typing.Any", "Any", "object"):
            if not text.startswith(token, index):
                continue
            before = text[index - 1] if index > 0 else ""
            after_index = index + len(token)
            after = text[after_index] if after_index < len(text) else ""
            before_is_identifier = bool(before) and (before.isalnum() or before == "_")
            after_is_identifier = bool(after) and (
                after.isalnum() or after in {"_", "."}
            )
            if before_is_identifier or after_is_identifier:
                continue
            return token, "t.JsonValue", after_index
        return None

    @staticmethod
    def _extract_square_bracket_content(text: str, open_index: int) -> tuple[str, int]:
        """Return the content and end offset for a square-bracket type expression."""
        depth = 0
        cursor = open_index
        while cursor < len(text):
            char = text[cursor]
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return text[open_index + 1 : cursor], cursor + 1
            cursor += 1
        return text[open_index + 1 :], len(text)

    def _rewrite_tuple_annotation(
        self, *, original_prefix: str, rewritten_content: str
    ) -> str:
        """Rewrite one ``tuple[...]`` annotation into the canonical ``t.*`` alias."""
        cls = FlextInfraRefactorTypingUnifierRewriteMixin
        parts = self._split_top_level_items(rewritten_content)
        if len(parts) == cls._VARIADIC_TUPLE_PARTS and parts[1] == "...":
            return f"t.VariadicTuple[{parts[0]}]"
        alias_name = cls._FIXED_TUPLE_ALIASES.get(len(parts))
        if alias_name is not None:
            return f"t.{alias_name}[{', '.join(parts)}]"
        prefix = "Tuple" if original_prefix.startswith("Tuple[") else "tuple"
        return f"{prefix}[{', '.join(parts)}]"

    @staticmethod
    def _split_top_level_items(text: str) -> list[str]:
        """Split a generic type-parameter list on top-level commas only."""
        items: list[str] = []
        start = 0
        depth = 0
        for index, char in enumerate(text):
            if char in "([{":
                depth += 1
            elif char in ")]}":
                if depth > 0:
                    depth -= 1
            elif char == "," and depth == 0:
                items.append(text[start:index].strip())
                start = index + 1
        items.append(text[start:].strip())
        return [item for item in items if item]


__all__: list[str] = ["FlextInfraRefactorTypingUnifierRewriteMixin"]
