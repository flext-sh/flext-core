"""Reusable docs rendering helpers exposed through ``u.Infra``."""

from __future__ import annotations

import fnmatch
import textwrap
from pathlib import Path
from typing import ClassVar

from flext_cli import u
from flext_infra import c, config, m, t


class FlextInfraUtilitiesDocsRender:
    """Rendering helpers for generated docs content."""

    _MARKDOWN_LINE_LENGTH: ClassVar[int] = 80

    @staticmethod
    def _wrap_markdown_line(line: str) -> t.SequenceOf[str]:
        """Wrap one generated prose line without changing Markdown structure."""
        if len(line) <= FlextInfraUtilitiesDocsRender._MARKDOWN_LINE_LENGTH:
            return (line,)
        stripped = line.lstrip()
        if (
            not stripped
            or stripped != line
            or stripped.startswith(("#", "|", ":::", "```", "<"))
        ):
            return (line,)
        prefix = ""
        body = line
        subsequent_indent = ""
        if line.startswith(("- ", "* ", "+ ")):
            prefix, body = line[:2], line[2:]
            subsequent_indent = "  "
        elif line.startswith("> "):
            prefix, body = line[:2], line[2:]
            subsequent_indent = prefix
        else:
            marker, separator, remainder = line.partition(" ")
            if separator and marker.endswith(".") and marker[:-1].isdigit():
                prefix, body = f"{marker} ", remainder
                subsequent_indent = " " * len(prefix)
        wrapper = textwrap.TextWrapper(
            width=FlextInfraUtilitiesDocsRender._MARKDOWN_LINE_LENGTH,
            initial_indent=prefix,
            subsequent_indent=subsequent_indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        return tuple(wrapper.wrap(body)) or (line,)

    @staticmethod
    def _render_markdown(lines: t.SequenceOf[str]) -> str:
        """Render generated Markdown with bounded prose and verbatim code blocks."""
        rendered: t.MutableSequenceOf[str] = []
        in_fence = False
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                rendered.append(line)
            elif in_fence:
                rendered.append(line)
            else:
                rendered.extend(FlextInfraUtilitiesDocsRender._wrap_markdown_line(line))
        return "\n".join(rendered)

    @staticmethod
    def _is_object_list(value: t.Infra.InfraValue | None) -> bool:
        """Type guard: narrow one infra value to a mutable sequence."""
        return isinstance(value, list)

    @staticmethod
    def as_string_sequence(data: t.JsonMapping, key: str) -> t.SequenceOf[str]:
        """Return one contract field as a normalized string sequence."""
        value = data.get(key)
        if not isinstance(value, list):
            return []
        return [str(entry) for entry in value]

    @staticmethod
    def _preview(values: t.SequenceOf[str], *, limit: int = 6) -> str:
        """Render one compact preview list with overflow summary."""
        normalized = [value for value in values if value]
        if not normalized:
            return "_none_"
        preview = ", ".join(f"`{value}`" for value in normalized[:limit])
        remaining = len(normalized) - limit
        return f"{preview} (+{remaining} more)" if remaining > 0 else preview

    @staticmethod
    def _escape_table_cell(value: str) -> str:
        """Escape Markdown table separators in one cell."""
        return value.replace("|", "\\|").strip()

    @staticmethod
    def _module_relative_doc_path(package_name: str, module_name: str) -> str:
        """Return the generated relative path for one module page."""
        relative = module_name.removeprefix(f"{package_name}.").replace(".", "/")
        return f"{relative}.md"

    @staticmethod
    def _resolve_governance_link(
        prefix: str, path: str, *, is_dir: bool = False
    ) -> str:
        """Return a resolvable governance link for README or project docs.

        READMEs render on GitHub and can use relative paths; generated
        ``docs/index.md`` pages are built by MkDocs with ``docs_dir`` isolation,
        so governance pointers must be absolute GitHub URLs.
        """
        if prefix.startswith(("http://", "https://")):
            kind = "tree" if is_dir else "blob"
            branch = "0.12.0-dev"
            for repo in config.Infra.codegen.make.docs.github_repos:
                if repo.organization == "flext-sh" and repo.repository == "flext":
                    branch = repo.branch
                    break
            return f"{prefix}/{kind}/{branch}/{path}"
        return f"{prefix}/{path}"

    @staticmethod
    def _exclude_plugin_lines(data: t.JsonMapping) -> t.SequenceOf[str]:
        """Render optional ``mkdocs-exclude`` plugin lines."""
        patterns = FlextInfraUtilitiesDocsRender.as_string_sequence(
            data, "exclude_docs"
        )
        if not patterns:
            return []
        return [
            "  - exclude:",
            "      glob:",
            *[f"        - {pattern}" for pattern in patterns],
        ]

    @staticmethod
    def _exclude_docs_lines(data: t.JsonMapping) -> t.SequenceOf[str]:
        """Render native ``exclude_docs`` lines for early MkDocs filtering.

        ``exclude_docs`` is gitignore-style (``pathspec.GitIgnoreSpec``): a bare
        ``README.md`` matches at every depth and silently drops legitimate
        ``docs/<section>/README.md`` pages that generated index pages link to
        (producing nav 404s). The rooted ``/README.md`` excludes only the
        docs-dir root README (the project README mirror), preserving nested
        section READMEs. [flext-3o9s nav404 fix]
        """
        patterns = list(
            dict.fromkeys([
                *FlextInfraUtilitiesDocsRender.as_string_sequence(data, "exclude_docs"),
                "/README.md",
            ])
        )
        return ["exclude_docs: |", *[f"  {pattern}" for pattern in patterns], ""]

    @staticmethod
    def docs_directive_page(title: str, dotted_path: str) -> str:
        """Return a mkdocstrings page for a module path."""
        return FlextInfraUtilitiesDocsRender._generated_page(
            title,
            [
                f"::: {dotted_path}",
                "    options:",
                "      show_root_heading: true",
                "      show_root_full_path: false",
                "      show_source: false",
            ],
        )

    _LINK_PREFIX_DOCS_INDEX: ClassVar[str] = c.Infra.GITHUB_REPO_URL
    """Link prefix for ``<project>/docs/index.md`` governance pointers.

    Uses the canonical GitHub URL so MkDocs can resolve the pointers during
    per-project builds; files under ``docs_dir`` cannot point outside the
    project docs tree with relative paths.
    """

    _LINK_PREFIX_README: ClassVar[str] = c.Infra.GITHUB_REPO_URL
    """Canonical governance URL shared by workspace and standalone READMEs."""

    @staticmethod
    def _public_surface_lines(scope: m.Infra.DocScope) -> t.SequenceOf[str]:
        """Return the canonical public-surface block — mkdocstrings autodoc.

        Replaces the prior hardcoded four-bullet summary (frozen snapshot of
        ``aliases`` / ``facades`` / ``public_symbols`` from contract data) with
        a live mkdocstrings directive that renders the package-level docstring
        + ``__all__`` exports at mkdocs build time. Real content from real
        code, never frozen — the user's "fake markdown" complaint.
        """
        return [
            f"::: {scope.package_name}",
            "    options:",
            "      members: false",
            "      show_root_heading: false",
            "      show_root_toc_entry: false",
            "      show_source: false",
        ]

    @staticmethod
    def _collection_rules_lines(
        scope: m.Infra.DocScope, *, link_prefix: str
    ) -> t.SequenceOf[str]:
        """Return a thin pointer to the canonical Collection Rules.

        SSOT: the actual content lives in ``flext/AGENTS.md`` §9 — duplicating
        it 33× per project (once for ``docs/index.md`` and once for
        ``README.md``) is the "fake markdown" the user flagged. Each project
        page now points back to the canonical source instead of carrying a
        copy. ``scope`` is preserved on the signature for symmetry with the
        other boilerplate helpers but is intentionally unused.
        """
        _ = scope
        agents_link = FlextInfraUtilitiesDocsRender._resolve_governance_link(
            link_prefix, "AGENTS.md"
        )
        return [
            "## Collection Rules",
            "",
            f"Read [`/flext/AGENTS.md`]({agents_link}) §9 — Agent Execution Pre-requisites — for the canonical pre-change checklist (parent FLEXT chain, Scope bootstrap, skill loading, zero-debt baseline,",
            "slot registry verification).",
        ]

    @staticmethod
    def _quality_gates_lines(*, link_prefix: str) -> t.SequenceOf[str]:
        """Return a thin pointer to the canonical Quality Gates surface.

        Why: flext-4p0t — flext-quality-gates skill path does not exist; route to
        make-check and AGENTS.md Make contract instead.
        """
        agents_link = FlextInfraUtilitiesDocsRender._resolve_governance_link(
            link_prefix, "AGENTS.md"
        )
        return [
            "## Quality Gates",
            "",
            (
                f"Canonical `make` verbs (`check`, `test`, `fmt WHAT=apply APPLY=Y`, "
                f"`val`, `docs`) — see [`/flext/AGENTS.md`]({agents_link}) `Build & Test` "
                f"and `Required Python quality gates`; selector routing is owned "
                f"universally by `config.AiHub.paths.agents_home`/"
                f"`skills/make-check/SKILL.md`."
            ),
        ]

    @staticmethod
    def _governance_pointer_lines(*, link_prefix: str) -> t.SequenceOf[str]:
        """Return a thin pointer to the canonical governance surface."""
        agents_link = FlextInfraUtilitiesDocsRender._resolve_governance_link(
            link_prefix, "AGENTS.md"
        )
        skills_link = FlextInfraUtilitiesDocsRender._resolve_governance_link(
            link_prefix, ".agents/skills/", is_dir=True
        )
        onboarding_link = FlextInfraUtilitiesDocsRender._resolve_governance_link(
            link_prefix, "docs/guides/onboarding.md"
        )
        governance_link = FlextInfraUtilitiesDocsRender._resolve_governance_link(
            link_prefix, "docs/GOVERNANCE.md"
        )
        return [
            "## Governance Pointer",
            "",
            f"- Engineering law: [`/flext/AGENTS.md`]({agents_link})",
            f"- Governance + ADRs: [`/flext/docs/GOVERNANCE.md`]({governance_link})",
            f"- Skills index: [`/flext/.agents/skills/`]({skills_link})",
            f"- Onboarding: [`/flext/docs/guides/onboarding.md`]({onboarding_link})",
        ]

    @staticmethod
    def docs_project_index(scope: m.Infra.DocScope, contract: t.JsonMapping) -> str:
        """Return the standard ``<project>/docs/index.md`` landing page."""
        data = contract
        version = str(data.get("version", "")).strip() or "unknown"
        description = str(data.get("description", "")).strip() or "_not declared_"
        link_prefix = FlextInfraUtilitiesDocsRender._LINK_PREFIX_DOCS_INDEX
        return FlextInfraUtilitiesDocsRender._generated_page(
            f"{scope.name} Documentation",
            [
                f"- Version: `{version}`",
                f"- Project class: `{scope.project_class}`",
                f"- Package: `{scope.package_name}`",
                f"- Description: {description}",
                "",
                "This project portal is generated from `pyproject.toml`, package exports, and real docstrings.",
                "",
                "## Start Here",
                "",
                "- [Guides](guides/README.md)",
                "- [API Reference](api-reference/README.md)",
                "- [Generated API Overview](api-reference/generated/overview.md)",
                "- [Generated Module Index](api-reference/generated/modules/index.md)",
                "",
                "## Public Surface Summary",
                "",
                *FlextInfraUtilitiesDocsRender._public_surface_lines(scope),
                "",
                *FlextInfraUtilitiesDocsRender._collection_rules_lines(
                    scope, link_prefix=link_prefix
                ),
                "",
                *FlextInfraUtilitiesDocsRender._quality_gates_lines(
                    link_prefix=link_prefix
                ),
                "",
                *FlextInfraUtilitiesDocsRender._governance_pointer_lines(
                    link_prefix=link_prefix
                ),
            ],
        )

    @staticmethod
    def docs_project_readme(scope: m.Infra.DocScope, contract: t.JsonMapping) -> str:
        """Return the canonical ``<project>/README.md`` (8-section structure).

        Auto-generated from ``pyproject.toml`` + package exports + the canonical
        FLEXT readme template (see ``readme-standardization`` skill). Output is
        fully deterministic — running the generator twice produces byte-identical
        content. Shares Collection Rules, Quality Gates, and Governance Pointer
        sections with ``docs_project_index`` via private helpers (no duplication).
        """
        data = contract
        version = str(data.get("version", "")).strip() or "unknown"
        description = str(data.get("description", "")).strip() or "_not declared_"
        facades = FlextInfraUtilitiesDocsRender.as_string_sequence(data, "facades")
        link_prefix = FlextInfraUtilitiesDocsRender._LINK_PREFIX_README
        return FlextInfraUtilitiesDocsRender._render_markdown([
            f"# {scope.name}",
            "",
            c.Infra.GENERATED_HEADER,
            "",
            f"**Version**: `{version}` | **Python**: 3.13+ | **Project class**: `{scope.project_class}`",
            "",
            (
                "> **Alpha (0.12.0).** This package is alpha quality. Every "
                "package in the workspace must be re-checked and re-validated "
                "at 0.12.0 before any promotion beyond alpha; treat interfaces "
                "as unstable."
            ),
            "",
            "## Purpose",
            "",
            description,
            "",
            "## Module Map",
            "",
            *FlextInfraUtilitiesDocsRender._public_surface_lines(scope),
            "",
            *FlextInfraUtilitiesDocsRender._collection_rules_lines(
                scope, link_prefix=link_prefix
            ),
            "",
            "## Operation Flow",
            "",
            "- Public surface: see [`docs/index.md`](docs/index.md) and [`docs/api-reference/README.md`](docs/api-reference/README.md).",
            "- Generated module overview: [`docs/api-reference/generated/overview.md`](docs/api-reference/generated/overview.md).",
            "- Settings env prefix: see project `pyproject.toml` `[tool.flext]` and `FlextSettings` ConfigDict.",
            "",
            "## Integration Points",
            "",
            "- Parent FLEXT chain: read this project's `pyproject.toml` `dependencies` array filtered by `flext-*`. The FLEXT cascade is encoded in the inheritance lists of the facade classes listed under Module Map above.",
            f"- Public extensions exposed by this project: {FlextInfraUtilitiesDocsRender._preview(facades)}.",
            "- Library abstraction boundaries: see AGENTS.md §2.7.",
            "",
            *FlextInfraUtilitiesDocsRender._quality_gates_lines(
                link_prefix=link_prefix
            ),
            "",
            *FlextInfraUtilitiesDocsRender._governance_pointer_lines(
                link_prefix=link_prefix
            ),
            "- Full project portal: [`docs/index.md`](docs/index.md).",
            "",
        ])

    @staticmethod
    def docs_guides_index(scope: m.Infra.DocScope) -> str:
        """Return a guides index that lists the guides the project really has.

        The index is generated, so every link it renders must resolve. Naming a
        curated guide the generator never writes produced a broken relative
        link in every project that had no such file.
        """
        guides_dir = scope.path / "docs/guides"
        entries = [
            f"- [{path.stem.replace('-', ' ').capitalize()}]({path.name})"
            for path in sorted(guides_dir.glob("*.md"))
            if path.name != "README.md"
        ]
        return FlextInfraUtilitiesDocsRender._render_markdown([
            f"# {scope.name} Guides",
            "",
            c.Infra.GENERATED_HEADER,
            "",
            "Curated operational guides live here. Keep API behavior in generated reference pages sourced from code and docstrings.",
            "",
            *entries,
            "- [Back to project docs](../index.md)",
            "- [API Reference](../api-reference/README.md)",
            "",
        ])

    @staticmethod
    def docs_api_readme(scope: m.Infra.DocScope, contract: t.JsonMapping) -> str:
        """Return the standard API readme for a project."""
        data = contract
        facades = FlextInfraUtilitiesDocsRender.as_string_sequence(data, "facades")
        modules = FlextInfraUtilitiesDocsRender.as_string_sequence(data, "modules")
        return FlextInfraUtilitiesDocsRender._render_markdown([
            f"# {scope.name} API Reference",
            "",
            c.Infra.GENERATED_HEADER,
            "",
            "This section is generated from public exports and real docstrings.",
            "",
            "## Source of Truth",
            "",
            "1. `pyproject.toml` metadata",
            f"2. `src/{scope.package_name}/__init__.py` exports",
            "3. Module docstrings",
            "4. Class and function docstrings",
            "",
            "## Generated Pages",
            "",
            "- [Overview](generated/overview.md)",
            "- [Public API](generated/public-api.md)",
            "- [Module Index](generated/modules/index.md)",
            "",
            "## Surface Summary",
            "",
            f"- Primary facades: {FlextInfraUtilitiesDocsRender._preview(facades)}",
            f"- Generated module pages: `{len(modules)}`",
            "",
            "Back to [project docs](../index.md).",
            "",
        ])

    @staticmethod
    def _generated_page(title: str, body: t.SequenceOf[str]) -> str:
        """Compose a generated markdown page with the canonical header + title.

        SSOT for the ``["# title", "", GENERATED_HEADER, "", ...body, ""]``
        layout shared by every ``docs_*_page``/``docs_*_index``/``docs_*_readme``
        renderer. Structural lines remain verbatim while prose is wrapped by
        the canonical renderer.
        """
        return FlextInfraUtilitiesDocsRender._render_markdown([
            f"# {title}",
            "",
            c.Infra.GENERATED_HEADER,
            "",
            *body,
            "",
        ])

    @staticmethod
    def _render_block(lines: t.SequenceOf[str]) -> str:
        """Pre-format a line sequence into a newline-terminated j2-safe block.

        Returns ``""`` when ``lines`` is empty so the template's substitution
        leaves no trailing artifacts. Otherwise joins with newlines and adds
        one trailing newline so the next macro sits on a fresh line.
        """
        return "\n".join([*lines, ""]) if lines else ""

    @staticmethod
    def _mkdocstrings_paths_block(paths: t.SequenceOf[str]) -> str:
        """Pre-format the optional mkdocstrings ``paths`` sub-block."""
        if not paths:
            return ""
        body = ["          paths:", *(f"            - {path}" for path in paths)]
        return "\n".join([*body, ""])

    @staticmethod
    def docs_project_mkdocs(
        scope: m.Infra.DocScope, contract: t.JsonMapping, modules: t.SequenceOf[str]
    ) -> str:
        """Return the managed mkdocs.yml for a project scope.

        Renders the canonical ``mkdocs_project.yml.j2`` template through the
        flext-cli facade shared with the workspace variant; theme, plugins,
        and validation macros live in the template, not Python.
        """
        _ = modules
        data = contract

        # NOTE (multi-agent, flext-p4s3.2 / agent: uv_overlay_owner): preserve one
        # typed context across the sole public template-rendering boundary.
        context = m.Infra.MkdocsProjectRenderContext(
            site_title=str(data.get("site_title", "")).strip() or scope.name,
            site_url=str(data.get("site_url", "")).strip() or c.Infra.GITHUB_REPO_URL,
            repo_url=str(data.get("repo_url", "")).strip() or c.Infra.GITHUB_REPO_URL,
            repo_name=c.Infra.GITHUB_REPO_NAME,
            scope_name=scope.name,
            exclude_docs_block=FlextInfraUtilitiesDocsRender._render_block(
                FlextInfraUtilitiesDocsRender._exclude_docs_lines(data)
            ),
            exclude_plugin_block=FlextInfraUtilitiesDocsRender._render_block(
                FlextInfraUtilitiesDocsRender._exclude_plugin_lines(data)
            ),
            mkdocstrings_paths_block=FlextInfraUtilitiesDocsRender._mkdocstrings_paths_block((
                "src",
            )),
        )
        template_path = (
            Path(__file__).resolve().parent.parent
            / "templates"
            / c.Infra.TEMPLATE_MKDOCS_PROJECT
        )
        rendered: str = t.Infra.STR_ADAPTER.validate_python(
            u.Cli.template_render(template_path, context).unwrap()
        )
        return rendered

    @staticmethod
    def docs_overview_page(scope: m.Infra.DocScope, contract: t.JsonMapping) -> str:
        """Return the generated overview page for a project API."""
        data = contract
        aliases = FlextInfraUtilitiesDocsRender._preview(
            FlextInfraUtilitiesDocsRender.as_string_sequence(data, "aliases"), limit=11
        )
        exports = FlextInfraUtilitiesDocsRender._preview(
            FlextInfraUtilitiesDocsRender.as_string_sequence(data, "public_symbols"),
            limit=10,
        )
        facades = FlextInfraUtilitiesDocsRender._preview(
            FlextInfraUtilitiesDocsRender.as_string_sequence(data, "facades"), limit=8
        )
        module_exports = FlextInfraUtilitiesDocsRender._preview(
            FlextInfraUtilitiesDocsRender.as_string_sequence(data, "module_exports"),
            limit=8,
        )
        modules = FlextInfraUtilitiesDocsRender.as_string_sequence(data, "modules")
        keywords = FlextInfraUtilitiesDocsRender._preview(
            FlextInfraUtilitiesDocsRender.as_string_sequence(data, "keywords"), limit=8
        )
        return FlextInfraUtilitiesDocsRender._render_markdown([
            f"# {(data.get('site_title', '') or scope.name)} API Overview",
            "",
            c.Infra.GENERATED_HEADER,
            "",
            f"- Package: `{scope.package_name}`",
            f"- Version: `{data.get('version', '')}`",
            f"- Description: {data.get('description', '') or '_not declared_'}",
            f"- Doc summary: {str(data.get('doc_summary', '')).strip() or '_not declared_'}",
            f"- Classifiers: {FlextInfraUtilitiesDocsRender._preview(FlextInfraUtilitiesDocsRender.as_string_sequence(data, 'classifiers'), limit=6)}",
            f"- Project class: `{scope.project_class}`",
            f"- Keywords: {keywords}",
            f"- Main facades: {facades}",
            f"- Alias exports: {aliases}",
            f"- Public symbol exports: {exports}",
            f"- Exported module shortcuts: {module_exports}",
            f"- Generated module pages: `{len(modules)}`",
            "",
            "## Next Pages",
            "",
            "- [Public API](public-api.md)",
            "- [Module Index](modules/index.md)",
            "",
        ])

    @staticmethod
    def docs_modules_index(scope: m.Infra.DocScope, modules: t.SequenceOf[str]) -> str:
        """Return the generated module index page for one project."""
        lines: t.MutableSequenceOf[str] = [
            f"# {scope.name} Module Index",
            "",
            c.Infra.GENERATED_HEADER,
            "",
            "These pages are generated from public modules and their docstrings.",
            "",
        ]
        if not modules:
            lines.extend(["_No public modules discovered._", ""])
            return FlextInfraUtilitiesDocsRender._render_markdown(lines)
        for module_name in modules:
            relative_path = FlextInfraUtilitiesDocsRender._module_relative_doc_path(
                scope.package_name, module_name
            )
            lines.append(f"- [{module_name}]({relative_path})")
        lines.append("")
        return FlextInfraUtilitiesDocsRender._render_markdown(lines)

    @staticmethod
    def docs_root_mkdocs(
        contract: t.JsonMapping, src_paths: t.SequenceOf[str] = ()
    ) -> str:
        """Return the managed mkdocs.yml for the repository root.

        Renders the canonical ``mkdocs_root.yml.j2`` template through the
        flext-cli facade shared with the per-project variant.
        """
        data = contract

        # NOTE (multi-agent, flext-p4s3.2 / agent: uv_overlay_owner): preserve one
        # typed context across the sole public template-rendering boundary.
        context = m.Infra.MkdocsRenderContext(
            site_title=str(data.get("site_title", "")).strip() or "FLEXT Workspace",
            site_url=str(data.get("site_url", "")).strip() or c.Infra.GITHUB_REPO_URL,
            repo_url=str(data.get("repo_url", "")).strip() or c.Infra.GITHUB_REPO_URL,
            repo_name=c.Infra.GITHUB_REPO_NAME,
            exclude_docs_block=FlextInfraUtilitiesDocsRender._render_block(
                FlextInfraUtilitiesDocsRender._exclude_docs_lines(data)
            ),
            exclude_plugin_block=FlextInfraUtilitiesDocsRender._render_block(
                FlextInfraUtilitiesDocsRender._exclude_plugin_lines(data)
            ),
            mkdocstrings_paths_block=FlextInfraUtilitiesDocsRender._mkdocstrings_paths_block(
                src_paths
            ),
        )
        template_path = (
            Path(__file__).resolve().parent.parent
            / "templates"
            / c.Infra.TEMPLATE_MKDOCS_ROOT
        )
        rendered: str = t.Infra.STR_ADAPTER.validate_python(
            u.Cli.template_render(template_path, context).unwrap()
        )
        return rendered

    @staticmethod
    def docs_root_overview_page(
        contract: t.JsonMapping,
        *,
        project_count: int,
        class_counts: t.MappingKV[str, int],
    ) -> str:
        """Return the generated root API overview page."""
        data = contract
        classes = (
            ", ".join(
                f"`{name}`={count}" for name, count in sorted(class_counts.items())
            )
            or "_none_"
        )
        return FlextInfraUtilitiesDocsRender._render_markdown([
            f"# {str(data.get('site_title', '')).strip() or 'FLEXT Workspace'} API Overview",
            "",
            c.Infra.GENERATED_HEADER,
            "",
            f"- Version: `{str(data.get('version', '')).strip() or 'unknown'}`",
            f"- Description: {str(data.get('description', '')).strip() or '_not declared_'}",
            f"- Governed projects: `{project_count}`",
            f"- Project classes: {classes}",
            "",
            "Generated from workspace discovery, `pyproject.toml`, public exports, and docstrings.",
            "",
            "## Next Pages",
            "",
            "- [Workspace Module Pages](projects/index.md)",
            "- [Project Catalog](../../projects/generated/catalog.md)",
            "",
        ])

    @staticmethod
    def docs_root_projects_index(entries: t.SequenceOf[t.StrMapping]) -> str:
        """Return the generated root index of per-project module pages."""
        lines: t.MutableSequenceOf[str] = [
            "# Workspace Module Pages",
            "",
            c.Infra.GENERATED_HEADER,
            "",
            "Each project renders one page per public module, driven by docstrings.",
            "",
        ]
        if not entries:
            lines.extend(["_No projects discovered._", ""])
            return FlextInfraUtilitiesDocsRender._render_markdown(lines)
        for entry in entries:
            lines.append(
                f"- [{entry['name']}]({entry['name']}/modules/index.md)"
                f" — `{entry['module_count']}` modules"
            )
        lines.append("")
        return FlextInfraUtilitiesDocsRender._render_markdown(lines)

    @staticmethod
    def docs_project_catalog_page(
        entries: t.SequenceOf[t.StrMapping],
        *,
        exclude_docs: t.SequenceOf[str] | None = None,
    ) -> str:
        """Return the generated workspace project catalog page."""
        exclude_patterns: t.StrSequence = tuple(exclude_docs or ())
        rows = [
            "| "
            + " | ".join([
                f"[{entry['name']}]({entry['api_page']})",
                FlextInfraUtilitiesDocsRender._escape_table_cell(
                    entry["project_class"]
                ),
                f"`{entry['package_name']}`",
                FlextInfraUtilitiesDocsRender._escape_table_cell(
                    entry["description"] or "_not declared_"
                ),
            ])
            + " |"
            for entry in entries
            if not any(
                fnmatch.fnmatch(
                    entry.get("api_page", "").removeprefix("../../"), pattern
                )
                for pattern in exclude_patterns
            )
        ]
        return FlextInfraUtilitiesDocsRender._render_markdown([
            "# FLEXT Project Catalog",
            "",
            c.Infra.GENERATED_HEADER,
            "",
            "Project links resolve to the generated root API reference for each governed FLEXT package.",
            "",
            "| project | class | package | description |",
            "|---|---|---|---|",
            *rows,
            "",
        ])


__all__: list[str] = ["FlextInfraUtilitiesDocsRender"]
