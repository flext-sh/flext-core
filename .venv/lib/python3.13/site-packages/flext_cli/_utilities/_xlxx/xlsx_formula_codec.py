"""OOXML storage codec for authored formula text."""

from __future__ import annotations

# mro-j47u (kimi): utilities consume local facades only, never private modules.
from flext_cli import c


class FlextCliUtilitiesXlsxFormulaCodec:
    """Encode authored formulas into their OOXML storage form."""

    # NOTE (multi-agent, mro-j2yt.1): OOXML stores functions introduced
    # after Excel 2007 with the _xlfn. prefix and conforming readers map
    # it back. Authored formulas keep canonical names; this codec owns the
    # single storage transformation at the external write boundary and
    # never rewrites text inside string literals.
    @classmethod
    def storage_formula(cls, formula: str) -> str:
        future = c.Cli.XLSX_FUTURE_FUNCTIONS
        prefix = c.Cli.XLSX_FUTURE_FUNCTION_PREFIX
        parts: list[str] = []
        index = 0
        length = len(formula)
        in_string = False
        while index < length:
            char = formula[index]
            if in_string:
                parts.append(char)
                if char == '"':
                    if index + 1 < length and formula[index + 1] == '"':
                        parts.append('"')
                        index += 2
                        continue
                    in_string = False
                index += 1
                continue
            if char == '"':
                in_string = True
                parts.append(char)
                index += 1
                continue
            if char.isalpha() or char == "_":
                end = index + 1
                while end < length and (formula[end].isalnum() or formula[end] in "._"):
                    end += 1
                token = formula[index:end]
                if token.upper() in future and end < length and formula[end] == "(":
                    parts.append(prefix + token)
                else:
                    parts.append(token)
                index = end
                continue
            parts.append(char)
            index += 1
        return "".join(parts)


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxFormulaCodec",)
