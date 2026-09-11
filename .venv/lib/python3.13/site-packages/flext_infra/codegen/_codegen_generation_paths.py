"""Path and publication helpers for lazy-init generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraCodegenGenerationPathsMixin:
    """Path and root-publication helper methods."""

    # flext-i6nq.10: Only canonical path/publication decisions remain here.
    @staticmethod
    def _is_module_or_package_export(attr_name: str) -> bool:
        """Return whether an entry exports a module or package name."""
        return not attr_name

    @staticmethod
    def _is_private_subpackage_source(module_path: str) -> bool:
        """Return whether a symbol's owner lives in a private subpackage."""
        return any(
            segment.startswith("_")
            and not segment.startswith("__")
            and segment not in c.Infra.LOCAL_INFERRED_SEGMENTS
            for segment in module_path.split(".")[1:]
        )

    @staticmethod
    def _should_publish_root_export(
        export_name: str, lazy_filtered: t.LazyAliasMap
    ) -> bool:
        """Return whether a root export belongs in the frozen ``__all__`` ABI."""
        if export_name in c.Infra.INFRA_ONLY_EXPORTS | c.Infra.PUBLISHED_ALL_EXCLUDE:
            return False
        target = lazy_filtered.get(export_name)
        if target is None:
            return True
        module_path, attr_name = target
        if FlextInfraCodegenGenerationPathsMixin._is_module_or_package_export(
            attr_name
        ):
            return export_name in c.Infra.PUBLIC_ROOT_MODULE_EXPORTS
        if not FlextInfraCodegenGenerationPathsMixin._is_private_subpackage_source(
            module_path
        ):
            return True
        return (
            export_name in c.Infra.ALIAS_NAMES
            or export_name in c.Infra.TEST_RUNTIME_ALIAS_TARGETS
        )

    @staticmethod
    def _is_root_namespace_package(current_pkg: str) -> bool:
        """Return whether a package name is a root namespace."""
        return bool(current_pkg) and "." not in current_pkg

    @staticmethod
    def _is_public_api_root_namespace(current_pkg: str) -> bool:
        """Return whether ``current_pkg`` owns a generated facade-root contract."""
        return FlextInfraCodegenGenerationPathsMixin._is_root_namespace_package(
            current_pkg
        ) and (
            current_pkg not in c.Infra.NON_PUBLIC_LAZY_ROOTS
            or current_pkg == c.Infra.DIR_TESTS
        )

    @staticmethod
    def _is_local_module(mod: str, root_name: str) -> bool:
        """Return whether ``mod`` is local to ``root_name``."""
        return (
            mod.startswith(".")
            or not root_name
            or mod.split(".", maxsplit=1)[0] == root_name
        )

    @staticmethod
    def _compact_lazy_module_path(current_pkg: str, mod: str) -> str:
        """Compact a lazy module path relative to ``current_pkg`` when valid."""
        if not current_pkg:
            return mod
        if mod.startswith("_"):
            return f".{mod}"
        if mod == current_pkg:
            return "."
        if mod.startswith(f"{current_pkg}."):
            return f".{mod.removeprefix(f'{current_pkg}.')}"
        root_pkg = current_pkg.split(".", maxsplit=1)[0]
        first_segment = mod.split(".", maxsplit=1)[0]
        internal_segments = frozenset(current_pkg.split(".")[1:])
        if (
            first_segment == root_pkg
            or internal_segments & c.Infra.LOCAL_INFERRED_SEGMENTS
        ):
            return mod
        if first_segment in internal_segments or (
            current_pkg == root_pkg
            and "." in mod
            and first_segment in c.Infra.LOCAL_INFERRED_SEGMENTS
        ):
            return f".{mod}"
        return mod

    @staticmethod
    def _normalize_type_checking_module_path(
        mod: str, local_package_root: str | None
    ) -> str:
        """Normalize local TYPE_CHECKING owners to package-relative imports."""
        if not local_package_root:
            return mod
        if mod.startswith("."):
            return mod
        if mod == local_package_root:
            return "."
        if mod.startswith(f"{local_package_root}."):
            return f".{mod.removeprefix(f'{local_package_root}.')}"
        root_pkg = local_package_root.split(".", maxsplit=1)[0]
        first_segment = mod.split(".", maxsplit=1)[0]
        internal_segments = frozenset(local_package_root.split(".")[1:])
        if (
            mod.startswith("_")
            or first_segment in internal_segments
            or (
                local_package_root == root_pkg
                and "." in mod
                and first_segment in c.Infra.LOCAL_INFERRED_SEGMENTS
            )
        ):
            return f".{mod}"
        return mod

    @staticmethod
    def _reject_noncanonical_type_checking_import(
        mod: str, local_package_root: str | None, items: t.StrPairSequence
    ) -> None:
        """Reject a relative TYPE_CHECKING import with no local package context.

        flext-udpm5: an absolute import that shares the current package's own
        project root but is not one of its descendants (an ancestor, sibling,
        or cousin package, e.g. flext-ldif's ``servers._oid`` importing
        ``servers.rfc``) is intentionally left absolute by
        ``_normalize_type_checking_module_path`` and accepted here rather
        than rejected: converting it to a relative import would require more
        than one leading dot, which the member project's own Ruff
        configuration (``ban-relative-imports = "parents"``) forbids. Only a
        descendant import lacking any local package context at all -- which
        cannot be resolved to any owner -- is rejected.
        """
        if mod.startswith(".") and not local_package_root:
            exports = ", ".join(name for name, _ in items)
            msg = (
                f"relative TYPE_CHECKING import {mod!r} has no local package "
                f"(exports: {exports})"
            )
            raise ValueError(msg)

    @staticmethod
    def _format_root_package_docstring(current_pkg: str) -> str:
        """Format a generated package docstring."""
        label = current_pkg.replace("_", " ").replace("-", " ").strip()
        package_name = " ".join(word.capitalize() for word in label.split())
        return f'"""{package_name} package."""'


__all__: list[str] = ["FlextInfraCodegenGenerationPathsMixin"]
