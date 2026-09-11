"""Generic helper mixin for Rope-backed refactors."""

from __future__ import annotations

from flext_infra._utilities._rope_bracket_balance import (
    FlextInfraUtilitiesRopeBracketBalanceMixin,
)
from flext_infra._utilities._rope_method_order import (
    FlextInfraUtilitiesRopeMethodOrderMixin,
)
from flext_infra.constants import c
from flext_infra.typings import t


class FlextInfraUtilitiesRopeHelpers(
    FlextInfraUtilitiesRopeBracketBalanceMixin, FlextInfraUtilitiesRopeMethodOrderMixin
):
    """Generic text, import-placement, and method-order helpers."""

    @staticmethod
    def get_module_level_assignments(source: str) -> t.StrPairSequence:
        """Return (name, value_str) for module-level simple assignments."""
        assignment_pattern = c.Infra.MODULE_ASSIGNMENT_RE
        results: list[t.StrPair] = []
        scope_depth = 0
        in_multiline_assignment = False
        current_name = ""
        current_value: list[str] = []
        open_brackets = 0

        for line in source.splitlines():
            stripped = line.strip()

            if not in_multiline_assignment:
                if stripped.startswith(("class ", "def ", "@")):
                    scope_depth += 1
                elif scope_depth > 0 and line and not line[0].isspace():
                    scope_depth = 0

            if scope_depth > 0:
                continue

            if in_multiline_assignment:
                current_value.append(stripped)
                open_brackets += (
                    stripped.count("(") + stripped.count("[") + stripped.count("{")
                )
                open_brackets -= (
                    stripped.count(")") + stripped.count("]") + stripped.count("}")
                )
                if open_brackets <= 0:
                    in_multiline_assignment = False
                    results.append((current_name, " ".join(current_value)))
                continue

            match = assignment_pattern.match(line)
            if match and not line[0].isspace():
                current_name = match.group(1)
                val_start = match.group(2).strip()
                open_brackets = (
                    val_start.count("(") + val_start.count("[") + val_start.count("{")
                )
                open_brackets -= (
                    val_start.count(")") + val_start.count("]") + val_start.count("}")
                )

                if open_brackets > 0:
                    in_multiline_assignment = True
                    current_value = [val_start]
                else:
                    results.append((current_name, val_start))

        return results

    @staticmethod
    def extract_definition(
        source: str, name: str, *, kind: str = "function"
    ) -> str | None:
        r"""Extract full def/class block by name using regex.

        Handles single-line signatures. Multi-line return-annotation
        signatures (``def foo() -> tuple[\n    A,\n]:``) are detected
        by a fallback bracket-balance scan that extends the regex match
        through any unclosed bracket groups.
        """
        if kind == "function":
            pattern = c.Infra.compile_function_def_block(name)
        elif kind == "class":
            pattern = c.Infra.compile_class_def_block(name)
        else:
            return None
        match = pattern.search(source)
        if match is None:
            return None
        block = match.group(0)
        return FlextInfraUtilitiesRopeHelpers._extend_block_through_open_brackets(
            source, block, match_end=match.end()
        ).rstrip("\n")

    @staticmethod
    def remove_definition(source: str, name: str, *, kind: str = "function") -> str:
        """Remove a top-level def/class from source."""
        if kind == "function":
            pattern = c.Infra.compile_function_def_remove(name)
        elif kind == "class":
            pattern = c.Infra.compile_class_def_remove(name)
        else:
            return source
        updated_source: str = pattern.sub("", source, count=1)
        return updated_source

    @staticmethod
    def append_to_class_body(source: str, class_name: str, block: str) -> str:
        """Append a block of code to an existing class body."""
        if not c.Infra.compile_class_header_search(class_name).search(source):
            return source.rstrip("\n") + f"\n\nclass {class_name}:\n{block}\n"
        lines = source.splitlines(keepends=True)
        in_class = False
        insert_idx = len(lines)
        class_indent = 0
        pass_idx: int | None = None
        only_placeholder_pass = True
        for index, line in enumerate(lines):
            stripped = line.lstrip()
            if not in_class:
                if stripped.startswith((f"class {class_name}", f"class {class_name}(")):
                    in_class = True
                    class_indent = len(line) - len(stripped) + 4
                continue
            if not line.strip():
                continue
            line_indent = len(line) - len(line.lstrip())
            if line_indent < class_indent and line.strip():
                insert_idx = index
                break
            if (
                line_indent == class_indent
                and stripped.strip() == "pass"
                and pass_idx is None
            ):
                pass_idx = index
                continue
            only_placeholder_pass = False
        if pass_idx is not None and only_placeholder_pass:
            del lines[pass_idx]
            if pass_idx < insert_idx:
                insert_idx -= 1
        lines.insert(insert_idx, block.rstrip("\n") + "\n\n")
        return "".join(lines)


__all__: list[str] = ["FlextInfraUtilitiesRopeHelpers"]
