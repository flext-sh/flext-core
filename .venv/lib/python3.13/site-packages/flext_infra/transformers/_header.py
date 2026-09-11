"""Safe Python module header parsing and canonical import injection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
import token
import tokenize
from dataclasses import dataclass
from typing import TYPE_CHECKING

from flext_infra import c

if TYPE_CHECKING:
    from flext_infra import t


@dataclass
class _HeaderSpan:
    """Byte offsets for the logical sections of a Python module header."""

    shebang_end: int = 0
    encoding_end: int = 0
    comments_end: int = 0
    docstring_end: int = 0
    last_import_end: int = 0

    @property
    def future_insert_offset(self) -> int:
        """Offset where ``from __future__ import annotations`` belongs.

        After shebang/encoding/comments and any module docstring.
        """
        return max(
            self.shebang_end, self.encoding_end, self.comments_end, self.docstring_end
        )

    @property
    def import_insert_offset(self) -> int:
        """Offset where a new regular import line belongs.

        After the last existing import; otherwise after the docstring.
        """
        if self.last_import_end:
            return self.last_import_end
        return max(
            self.shebang_end, self.encoding_end, self.comments_end, self.docstring_end
        )


@dataclass
class _HeaderInfo:
    """Structural summary of a module header."""

    has_future_annotations: bool
    aliases: frozenset[str]
    span: _HeaderSpan


def ensure_future_annotations(source: str) -> str:
    """Return source with exactly one ``from __future__ import annotations``.

    The future import is placed after any module docstring and after any
    shebang/encoding/comment header. Existing duplicates are removed.
    """
    normalized = _remove_future_annotations_lines(source)
    body = normalized.splitlines(keepends=True)
    info = _parse_header(normalized)
    offset = info.span.future_insert_offset
    line_index = normalized[:offset].count("\n")
    future_line = f"{c.Infra.FUTURE_ANNOTATIONS}\n"
    if info.span.docstring_end:
        while line_index < len(body) and not body[line_index].strip():
            del body[line_index]
        insert_lines = ["\n", future_line]
        if line_index < len(body):
            insert_lines.append("\n")
        body[line_index:line_index] = insert_lines
    else:
        body.insert(line_index, future_line)
    return "".join(body)


def ensure_alias_import(source: str, module: str, alias: str) -> str:
    """Inject ``from <module> import <alias>`` when the alias is actually used."""
    if not alias or not alias_used(source, alias):
        return source
    if has_alias_import(source, alias) or alias_locally_bound(source, alias):
        return source
    info = _parse_header(source)
    offset = info.span.import_insert_offset
    line = f"from {module} import {alias}\n"
    if offset < len(source) and not source[offset:].startswith("\n"):
        line = f"{line}\n"
    return f"{source[:offset]}{line}{source[offset:]}"


def alias_used(source: str, alias: str) -> bool:
    """Return whether ``alias`` is used as a standalone dotted identifier."""
    return (
        c.Infra.compile(rf"\b{c.Infra.escape(alias)}\.(?![0-9])").search(source)
        is not None
    )


def has_alias_import(source: str, alias: str) -> bool:
    """Return whether ``alias`` is already bound by a ``from`` import."""
    info = _parse_header(source)
    return alias in info.aliases


def alias_locally_bound(source: str, alias: str) -> bool:
    """Return whether ``alias`` is bound by a module-level definition/assignment.

    Facade-root modules (``typings.py``, ``constants.py``, ...) bind their
    canonical alias directly (``t = <Project>Types``). Injecting
    ``from <pkg> import <alias>`` into such a module would shadow that binding
    and raise F811 — a self-package import. Suppress the injection whenever the
    module already owns the alias through a top-level assignment, ``class`` or
    ``def``.
    """
    escaped = c.Infra.escape(alias)
    pattern = c.Infra.compile(
        rf"^(?:{escaped}\s*(?::[^=\n]+)?=(?!=)|(?:class|def)\s+{escaped}\b)",
        multiline=True,
    )
    return pattern.search(source) is not None


def _remove_future_annotations_lines(source: str) -> str:
    """Strip every ``from __future__ import annotations`` line."""
    lines: list[str] = []
    for line in source.splitlines(keepends=True):
        if line.strip() == c.Infra.FUTURE_ANNOTATIONS:
            continue
        lines.append(line)
    return "".join(lines)


def _parse_header(source: str) -> _HeaderInfo:
    """Parse the module header using the stdlib ``tokenize`` module."""
    aliases: set[str] = set()
    span = _HeaderSpan()

    try:
        tokens = list(tokenize.tokenize(io.BytesIO(source.encode()).readline))
    except tokenize.TokenError:
        return _HeaderInfo(False, frozenset(), span)

    if not tokens:
        return _HeaderInfo(False, frozenset(), span)

    span.shebang_end = _shebang_end(source)
    span.encoding_end = _encoding_end(source, span.shebang_end)
    span.comments_end = _leading_comments_end(source, span.encoding_end)

    has_future_annotations = False
    last_import_end = 0
    docstring_end = 0
    seen_non_header_token = False

    for index, tok in enumerate(tokens):
        if tok.type in {
            token.ENCODING,
            token.NL,
            token.NEWLINE,
            token.COMMENT,
            token.ENDMARKER,
        }:
            continue
        if (
            tok.type == token.STRING
            and not seen_non_header_token
            and docstring_end == 0
        ):
            docstring_end = _line_end_offset(source, tok.end)
            continue
        seen_non_header_token = True
        if tok.type != token.NAME or tok.string != "from":
            continue
        if index + 2 >= len(tokens):
            continue
        module_tok = tokens[index + 1]
        if module_tok.type != token.NAME:
            continue
        import_tok = tokens[index + 2]
        if import_tok.type != token.NAME or import_tok.string != "import":
            continue
        module_name = module_tok.string
        import_end = _find_import_line_end(source, tokens, index)
        if module_name == "__future__":
            if _imports_name(tokens, index, "annotations"):
                has_future_annotations = True
            last_import_end = max(last_import_end, import_end)
            continue
        aliases.update(_extract_imported_aliases(tokens, index))
        last_import_end = max(last_import_end, import_end)

    span.docstring_end = docstring_end
    span.last_import_end = last_import_end
    return _HeaderInfo(has_future_annotations, frozenset(aliases), span)


def _shebang_end(source: str) -> int:
    """Return byte offset after a shebang line, or 0."""
    if source.startswith("#!"):
        newline = source.find("\n")
        return len(source) if newline == -1 else newline + 1
    return 0


def _encoding_end(source: str, start: int) -> int:
    """Return byte offset after an encoding cookie, or start."""
    if start >= len(source):
        return start
    head = source[start:]
    match = c.Infra.ENCODING_COOKIE_RE.match(head)
    if match is None:
        return start
    line_end = head.find("\n")
    return start + (len(head) if line_end == -1 else line_end + 1)


def _leading_comments_end(source: str, start: int) -> int:
    """Return byte offset at the first non-comment, non-blank line.

    Blank lines are consumed only when they follow a leading comment block,
    matching the canonical placement of ``from __future__ import annotations``.
    """
    index = start
    saw_comment = False
    while index < len(source):
        line_end = source.find("\n", index)
        line = source[index:] if line_end == -1 else source[index:line_end]
        stripped = line.strip()
        if stripped.startswith("#"):
            saw_comment = True
            index = len(source) if line_end == -1 else line_end + 1
            continue
        if not stripped and saw_comment:
            index = len(source) if line_end == -1 else line_end + 1
            continue
        break
    return index


def _find_import_line_end(
    source: str, tokens: t.SequenceOf[tokenize.TokenInfo], from_index: int
) -> int:
    """Return the byte offset just after the import statement starting at from_index."""
    for tok in tokens[from_index:]:
        if tok.type in {token.NEWLINE, token.ENDMARKER}:
            return _position_offset(source, tok.end)
    return 0


def _line_end_offset(source: str, position: tuple[int, int]) -> int:
    """Return the source offset after the line containing ``position``."""
    offset = _position_offset(source, position)
    newline = source.find("\n", offset)
    return len(source) if newline == -1 else newline + 1


def _position_offset(source: str, position: tuple[int, int]) -> int:
    """Convert a ``tokenize`` line/column position into a source offset."""
    line, column = position
    if line <= 1:
        return min(column, len(source))
    offset = 0
    for _line_number in range(1, line):
        newline = source.find("\n", offset)
        if newline == -1:
            return len(source)
        offset = newline + 1
    return min(offset + column, len(source))


def _imports_name(
    tokens: t.SequenceOf[tokenize.TokenInfo], from_index: int, name: str
) -> bool:
    """Return whether the ``from ... import (...)`` binds ``name``."""
    return name in _extract_imported_aliases(tokens, from_index)


def _extract_imported_aliases(
    tokens: t.SequenceOf[tokenize.TokenInfo], from_index: int
) -> frozenset[str]:
    """Return local names bound by a ``from`` import statement."""
    aliases: set[str] = set()
    index = from_index + 3  # skip 'from' <module> 'import'
    while index < len(tokens):
        tok = tokens[index]
        if tok.type in {token.NEWLINE, token.ENDMARKER}:
            break
        if tok.type == token.OP and tok.string in {"(", ")"}:
            index += 1
            continue
        if tok.type == token.NAME:
            alias = tok.string
            if (
                index + 2 < len(tokens)
                and tokens[index + 1].type == token.NAME
                and tokens[index + 1].string == "as"
            ):
                alias = tokens[index + 2].string
                index += 2
            aliases.add(alias)
        index += 1
    return frozenset(aliases)


__all__: list[str] = [
    "alias_locally_bound",
    "alias_used",
    "ensure_alias_import",
    "ensure_future_annotations",
    "has_alias_import",
]
