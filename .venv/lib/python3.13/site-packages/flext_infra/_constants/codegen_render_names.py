"""Template name constants for the codegen package."""

from __future__ import annotations

from typing import Final


class FlextInfraConstantsCodegenRenderNames:
    """Template filename constants consumed by codegen renderers."""

    TEMPLATE_ROOT_INIT: Final[str] = "lazy_init_root.py.j2"
    "Public project-root ``__init__.py`` consuming owned lazy metadata."
    TEMPLATE_STATIC_INIT: Final[str] = "static_package_init.py.j2"
    "Non-root package ``__init__.py`` with explicit static reexports."
    TEMPLATE_VERSION_FILE: Final[str] = "version_file.py.j2"
    "Per-project ``__version__.py`` template."
    TEMPLATE_MKDOCS_PROJECT: Final[str] = "mkdocs_project.yml.j2"
    "Per-project ``mkdocs.yml`` template."
    TEMPLATE_MKDOCS_ROOT: Final[str] = "mkdocs_root.yml.j2"
    "Workspace ``mkdocs.yml`` template."


__all__: list[str] = ["FlextInfraConstantsCodegenRenderNames"]
