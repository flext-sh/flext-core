"""Small Rope-driven text patch helpers for governed module aliases."""

from __future__ import annotations


class FlextInfraUtilitiesRopeModulePatch:
    """String-level patch helpers driven by Rope-discovered module rules."""

    @classmethod
    def ensure_runtime_alias(cls, source: str, *, alias: str, target_name: str) -> str:
        """Return source with one canonical runtime alias guaranteed."""
        updated = cls._ensure_alias_line(source, alias=alias, target_name=target_name)
        return cls._ensure_all_entry(updated, name=alias)

    @staticmethod
    def _ensure_alias_line(source: str, *, alias: str, target_name: str) -> str:
        """Ensure alias line."""
        target_line = f"{alias} = {target_name}"
        lines = source.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith(f"{alias} ="):
                continue
            if stripped == target_line:
                return source
            lines[index] = target_line
            return "\n".join(lines) + ("\n" if source.endswith("\n") else "")
        if not source.endswith("\n"):
            source = f"{source}\n"
        spacer = "" if source.endswith("\n\n") else "\n"
        return f"{source}{spacer}{target_line}\n"

    @classmethod
    def _ensure_all_entry(cls, source: str, *, name: str) -> str:
        """Ensure all entry."""
        quoted_name = f'"{name}"'
        if quoted_name in source or f"'{name}'" in source:
            return source
        lines = source.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("__all__"):
                continue
            if "[" in line and "]" in line:
                return cls._update_single_line_all(
                    source, line_index=index, entry=quoted_name
                )
            return cls._update_block_all(lines, start=index, entry=quoted_name)
        return source

    @staticmethod
    def _update_single_line_all(source: str, *, line_index: int, entry: str) -> str:
        """Update single line all."""
        lines = source.splitlines()
        line = lines[line_index]
        closing_index = line.rfind("]")
        prefix = line[:closing_index].rstrip()
        if prefix.endswith("["):
            lines[line_index] = f"{prefix}{entry}]"
        else:
            lines[line_index] = f"{prefix}, {entry}]"
        return "\n".join(lines) + ("\n" if source.endswith("\n") else "")

    @staticmethod
    def _update_block_all(lines: list[str], *, start: int, entry: str) -> str:
        """Update block all."""
        for index in range(start, len(lines)):
            if "]" not in lines[index]:
                continue
            indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
            lines.insert(index, f"{indent}    {entry},")
            return "\n".join(lines) + "\n"
        return "\n".join(lines) + "\n"


__all__: list[str] = ["FlextInfraUtilitiesRopeModulePatch"]
