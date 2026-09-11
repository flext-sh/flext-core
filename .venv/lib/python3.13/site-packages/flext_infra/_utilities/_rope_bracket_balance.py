"""Bracket-balance scanning for multi-line def/class block extraction — extracted."""

from __future__ import annotations

import io
import tokenize


class FlextInfraUtilitiesRopeBracketBalanceMixin:
    """Compute net bracket depth to extend regex-captured blocks across newlines.

    Composed into FlextInfraUtilitiesRopeHelpers via inheritance; ``extract_definition``
    resolves ``_extend_block_through_open_brackets`` through the facade FLEXT.
    """

    @staticmethod
    def _extend_block_through_open_brackets(
        source: str, block: str, *, match_end: int
    ) -> str:
        r"""Extend ``block`` when its regex capture ends mid-bracket-group.

        The existing regex terminates on the first column-0 non-empty line; a
        multi-line signature like ``-> tuple[\n    A,\n]:`` therefore cuts
        off at the unindented ``]:`` line. When the captured block has
        unbalanced brackets, consume further lines until balance reaches 0.
        """
        cls = FlextInfraUtilitiesRopeBracketBalanceMixin
        balance = cls._bracket_balance_total(block)
        if balance <= 0:
            return block
        remaining = source[match_end:]
        additional = 0
        for line in remaining.splitlines(keepends=True):
            additional += len(line)
            balance += cls.bracket_balance_line(line)
            if balance <= 0:
                break
        extended = block + source[match_end : match_end + additional]
        tail_start = match_end + additional
        tail = source[tail_start:]
        for line in tail.splitlines(keepends=True):
            stripped = line.strip()
            if not stripped or line.startswith((" ", "\t")):
                extended += line
                continue
            break
        return extended

    @staticmethod
    def _bracket_balance_total(text: str) -> int:
        """Bracket balance total."""
        total = 0
        for line in text.splitlines():
            total += FlextInfraUtilitiesRopeBracketBalanceMixin.bracket_balance_line(
                line
            )
        return total

    @staticmethod
    def bracket_balance_line(line: str) -> int:
        """Return net bracket depth delta of one line (strings + ``#`` comments ignored)."""
        try:
            return sum(
                1 if token.string in "([{" else -1
                for token in tokenize.generate_tokens(io.StringIO(line).readline)
                if token.type == tokenize.OP and token.string in "()[]{}"
            )
        except tokenize.TokenError:
            return FlextInfraUtilitiesRopeBracketBalanceMixin._fallback_bracket_balance_line(
                line
            )

    @staticmethod
    def _fallback_bracket_balance_line(line: str) -> int:
        """Approximate bracket balance for incomplete lines that ``tokenize`` rejects."""
        balance = 0
        in_single_quote = False
        in_double_quote = False
        escaped = False
        for char in line:
            if escaped:
                escaped = False
                continue
            if char == "\\" and (in_single_quote or in_double_quote):
                escaped = True
                continue
            if in_single_quote:
                if char == "'":
                    in_single_quote = False
                continue
            if in_double_quote:
                if char == '"':
                    in_double_quote = False
                continue
            if char == "#":
                break
            if char == "'":
                in_single_quote = True
                continue
            if char == '"':
                in_double_quote = True
                continue
            if char in "([{":
                balance += 1
            elif char in ")]}":
                balance -= 1
        return balance


__all__: list[str] = ["FlextInfraUtilitiesRopeBracketBalanceMixin"]
