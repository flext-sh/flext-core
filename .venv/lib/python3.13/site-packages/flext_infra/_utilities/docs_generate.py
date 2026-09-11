"""Generation helpers for docs services."""

from __future__ import annotations

import re
from pathlib import Path

from flext_cli import u
from flext_infra._utilities.docs import FlextInfraUtilitiesDocs
from flext_infra._utilities.docs_api import FlextInfraUtilitiesDocsApi
from flext_infra._utilities.docs_contract import FlextInfraUtilitiesDocsContract
from flext_infra._utilities.docs_render import FlextInfraUtilitiesDocsRender
from flext_infra import config
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.typings import t


class FlextInfraUtilitiesDocsGenerate:
    """Reusable generation helpers exposed through ``u.Infra``."""

    @staticmethod
    def _module_names(scope: m.Infra.DocScope) -> list[str]:
        """Return config-owned public API module names for one distribution."""
        declared = config.Infra.codegen.make.docs.api_modules.get(scope.name, ())
        return [f"{scope.package_name}.{module}" for module in declared]

    @staticmethod
    def _prune_generated_tree(
        root: Path, expected: t.SequenceOf[Path], *, apply: bool
    ) -> t.SequenceOf[m.Infra.GeneratedFile]:
        """Prune stale files from one tool-owned generated tree."""
        if not root.exists():
            return []
        expected_paths = {path.resolve() for path in expected}
        removed: t.MutableSequenceOf[m.Infra.GeneratedFile] = []
        for path in sorted(root.rglob("*.md")):
            if path.resolve() in expected_paths:
                continue
            if apply:
                path.unlink(missing_ok=True)
            removed.append(
                m.Infra.GeneratedFile(path=path.as_posix(), changed=True, written=apply)
            )
        return removed

    @staticmethod
    def docs_project_generated_files(
        scope: m.Infra.DocScope, *, apply: bool
    ) -> t.SequenceOf[m.Infra.GeneratedFile]:
        """Generate the managed docs artifacts for one FLEXT project."""
        contract = FlextInfraUtilitiesDocsApi.public_contract(
            scope.path, scope.package_name
        )
        module_names = FlextInfraUtilitiesDocsGenerate._module_names(scope)
        expected_generated: t.MutableSequenceOf[Path] = [
            scope.path / "docs/api-reference/generated/overview.md",
            scope.path / "docs/api-reference/generated/public-api.md",
            scope.path / "docs/api-reference/generated/modules/index.md",
        ]
        files: t.MutableSequenceOf[m.Infra.GeneratedFile] = [
            FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                scope.path / "README.md",
                FlextInfraUtilitiesDocsRender.docs_project_readme(scope, contract),
                apply=apply,
            ),
            FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                scope.path / "docs/index.md",
                FlextInfraUtilitiesDocsRender.docs_project_index(scope, contract),
                apply=apply,
            ),
            FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                scope.path / "docs/guides/README.md",
                FlextInfraUtilitiesDocsRender.docs_guides_index(scope),
                apply=apply,
            ),
            FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                scope.path / "docs/api-reference/README.md",
                FlextInfraUtilitiesDocsRender.docs_api_readme(scope, contract),
                apply=apply,
            ),
            FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                scope.path / "mkdocs.yml",
                FlextInfraUtilitiesDocsRender.docs_project_mkdocs(
                    scope, contract, module_names
                ),
                apply=apply,
            ),
            FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                scope.path / "docs/api-reference/generated/modules/index.md",
                FlextInfraUtilitiesDocsRender.docs_modules_index(scope, module_names),
                apply=apply,
            ),
            FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                scope.path / "docs/api-reference/generated/overview.md",
                FlextInfraUtilitiesDocsRender.docs_overview_page(scope, contract),
                apply=apply,
            ),
            FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                scope.path / "docs/api-reference/generated/public-api.md",
                FlextInfraUtilitiesDocsRender.docs_directive_page(
                    f"{scope.name} Public API", scope.package_name
                ),
                apply=apply,
            ),
        ]
        for module_name in module_names:
            relative = module_name.removeprefix(f"{scope.package_name}.").replace(
                ".", "/"
            )
            expected_generated.append(
                scope.path / "docs/api-reference/generated/modules" / f"{relative}.md"
            )
            files.append(
                FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                    scope.path
                    / "docs/api-reference/generated/modules"
                    / f"{relative}.md",
                    FlextInfraUtilitiesDocsRender.docs_directive_page(
                        module_name, module_name
                    ),
                    apply=apply,
                )
            )
        files.extend(
            FlextInfraUtilitiesDocsGenerate._prune_generated_tree(
                scope.path / "docs/api-reference/generated",
                expected_generated,
                apply=apply,
            )
        )
        return files

    @staticmethod
    def docs_project_guides_files(
        scope: m.Infra.DocScope, *, repository_root: Path, apply: bool
    ) -> t.SequenceOf[m.Infra.GeneratedFile]:
        """Return project guide files managed by generation.

        Guide propagation is intentionally disabled; curated guides stay local.
        """
        _ = scope
        _ = repository_root
        _ = apply
        return []

    @staticmethod
    def docs_project_mkdocs_files(
        scope: m.Infra.DocScope, *, apply: bool
    ) -> t.SequenceOf[m.Infra.GeneratedFile]:
        """Return the managed mkdocs settings file when it does not exist yet."""
        path = scope.path / "mkdocs.yml"
        if path.exists():
            return []
        contract = FlextInfraUtilitiesDocsApi.public_contract(
            scope.path, scope.package_name
        )
        module_names = FlextInfraUtilitiesDocsGenerate._module_names(scope)
        return [
            FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                path,
                FlextInfraUtilitiesDocsRender.docs_project_mkdocs(
                    scope, contract, module_names
                ),
                apply=apply,
                overwrite=False,
            )
        ]

    @staticmethod
    def docs_root_generated_files(
        repository_root: Path, *, apply: bool, projects: t.StrSequence | None = None
    ) -> t.SequenceOf[m.Infra.GeneratedFile]:
        """Generate root workspace docs artifacts from discovered FLEXT projects.

        The root site is an AGGREGATE of every workspace project, so scope
        discovery always enumerates all projects: honoring a ``projects``
        filter here would produce a partial aggregate whose prune step
        deletes the pages of every filtered-out project (flext-o6h5 incident).
        """
        _ = projects
        workspace_contract = FlextInfraUtilitiesDocsContract.docs_workspace_contract(
            repository_root
        )
        exclude_docs = FlextInfraUtilitiesDocsRender.as_string_sequence(
            workspace_contract, "exclude_docs"
        )
        scopes_result = FlextInfraUtilitiesDocs.build_scopes(
            repository_root, None, c.Infra.DEFAULT_DOCS_OUTPUT_DIR
        )
        scopes = (
            [scope for scope in scopes_result.value if scope.name != c.Infra.RK_ROOT]
            if scopes_result.success
            else []
        )
        catalog_entries: t.MutableSequenceOf[dict[str, str]] = []
        class_counts: dict[str, int] = {}
        # flext-o6h5 (agent: kimi) — root site aggregates per-project module
        # pages: module names come from the already-loaded public contract
        # and src paths feed the mkdocstrings resolution block.
        scope_modules: dict[str, list[str]] = {}
        src_paths: t.MutableSequenceOf[str] = []
        for scope in scopes:
            class_counts[scope.project_class] = (
                class_counts.get(scope.project_class, 0) + 1
            )
            project_contract = FlextInfraUtilitiesDocsApi.public_contract(
                scope.path, scope.package_name
            )
            scope_modules[scope.name] = FlextInfraUtilitiesDocsGenerate._module_names(
                scope
            )
            src_dir = scope.path / "src"
            if src_dir.is_dir():
                src_paths.append(src_dir.relative_to(repository_root).as_posix())
            catalog_entries.append({
                "name": scope.name,
                "project_class": scope.project_class,
                "package_name": scope.package_name,
                "description": str(project_contract.get("description", "")).strip(),
                "api_page": f"../../api-reference/generated/{scope.name}.md",
            })
        expected_api_generated: t.MutableSequenceOf[Path] = [
            repository_root / "docs/api-reference/generated/overview.md"
        ]
        expected_project_generated: t.MutableSequenceOf[Path] = [
            repository_root / "docs/projects/generated/catalog.md"
        ]
        files: t.MutableSequenceOf[m.Infra.GeneratedFile] = [
            FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                repository_root / "mkdocs.yml",
                FlextInfraUtilitiesDocsRender.docs_root_mkdocs(
                    workspace_contract, src_paths
                ),
                apply=apply,
            ),
            FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                repository_root / "docs/api-reference/generated/overview.md",
                FlextInfraUtilitiesDocsRender.docs_root_overview_page(
                    workspace_contract,
                    project_count=len(scopes),
                    class_counts=class_counts,
                ),
                apply=apply,
            ),
            FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                repository_root / "docs/projects/generated/catalog.md",
                FlextInfraUtilitiesDocsRender.docs_project_catalog_page(
                    catalog_entries, exclude_docs=exclude_docs
                ),
                apply=apply,
            ),
        ]
        projects_index_entries: t.MutableSequenceOf[dict[str, str]] = []
        for scope in scopes:
            expected_api_generated.append(
                repository_root / "docs/api-reference/generated" / f"{scope.name}.md"
            )
            files.append(
                FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                    repository_root
                    / "docs/api-reference/generated"
                    / f"{scope.name}.md",
                    FlextInfraUtilitiesDocsRender.docs_directive_page(
                        f"{scope.name} Public API", scope.package_name
                    ),
                    apply=apply,
                )
            )
            # flext-o6h5 (agent: kimi) — per-project module pages reuse the
            # exact project-scope renderers (docs_modules_index +
            # docs_directive_page); index lives inside modules/ so relative
            # links resolve identically to the project-scope layout.
            module_names = scope_modules.get(scope.name, [])
            modules_root = (
                repository_root
                / "docs/api-reference/generated/projects"
                / scope.name
                / "modules"
            )
            expected_api_generated.append(modules_root / "index.md")
            files.append(
                FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                    modules_root / "index.md",
                    FlextInfraUtilitiesDocsRender.docs_modules_index(
                        scope, module_names
                    ),
                    apply=apply,
                )
            )
            for module_name in module_names:
                relative = module_name.removeprefix(f"{scope.package_name}.").replace(
                    ".", "/"
                )
                module_path = modules_root / f"{relative}.md"
                expected_api_generated.append(module_path)
                files.append(
                    FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                        module_path,
                        FlextInfraUtilitiesDocsRender.docs_directive_page(
                            module_name, module_name
                        ),
                        apply=apply,
                    )
                )
            projects_index_entries.append({
                "name": scope.name,
                "module_count": str(len(module_names)),
            })
        projects_index_path = (
            repository_root / "docs/api-reference/generated/projects/index.md"
        )
        expected_api_generated.append(projects_index_path)
        files.append(
            FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                projects_index_path,
                FlextInfraUtilitiesDocsRender.docs_root_projects_index(
                    projects_index_entries
                ),
                apply=apply,
            )
        )
        files.extend(
            FlextInfraUtilitiesDocsGenerate._prune_generated_tree(
                repository_root / "docs/api-reference/generated",
                expected_api_generated,
                apply=apply,
            )
        )
        files.extend(
            FlextInfraUtilitiesDocsGenerate._prune_generated_tree(
                repository_root / "docs/projects/generated",
                expected_project_generated,
                apply=apply,
            )
        )
        return files

    @staticmethod
    def docs_sanitize_scope_fences(
        scope: m.Infra.DocScope, *, apply: bool
    ) -> t.SequenceOf[m.Infra.GeneratedFile]:
        """Remove unsupported ``notest`` qualifiers from code fence info lines."""
        changed: t.MutableSequenceOf[m.Infra.GeneratedFile] = []
        docs_root = scope.path / "docs"
        if not docs_root.exists():
            return changed
        for path in sorted(docs_root.rglob("*.md")):
            content = path.read_text(encoding=c.Cli.ENCODING_DEFAULT)
            sanitized = c.Infra.FENCE_NOTEST_RE.sub(r"```\1", content)
            if sanitized == content:
                continue
            changed.append(
                FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                    path, sanitized, apply=apply
                )
            )
        return changed

    @staticmethod
    def docs_generate_scope(
        scope: m.Infra.DocScope,
        *,
        apply: bool,
        repository_root: Path,
        projects: t.StrSequence | None = None,
    ) -> m.Infra.DocsPhaseReport:
        """Generate one scope and persist the standard reports."""
        files: t.MutableSequenceOf[m.Infra.GeneratedFile] = list(
            FlextInfraUtilitiesDocsGenerate.docs_root_generated_files(
                repository_root, apply=apply, projects=projects
            )
            if scope.name == c.Infra.RK_ROOT
            else FlextInfraUtilitiesDocsGenerate.docs_project_generated_files(
                scope, apply=apply
            )
        )
        files.extend(
            FlextInfraUtilitiesDocsGenerate.docs_sanitize_scope_fences(
                scope, apply=apply
            )
        )
        changed = u.count(files, lambda item: item.changed)
        generated = u.count(files, lambda item: item.written)
        files_payload: t.JsonList = [
            {"path": item.path, "changed": item.changed, "written": item.written}
            for item in files
        ]
        summary_payload = t.Cli.JSON_MAPPING_ADAPTER.validate_python({
            c.Infra.RK_SUMMARY: {
                c.Infra.RK_SCOPE: scope.name,
                "changed": changed,
                "generated": generated,
                "apply": apply,
            },
            "files": files_payload,
        })
        _ = u.Cli.json_write(
            scope.report_dir / "generate-summary.json", summary_payload
        )
        _ = FlextInfraUtilitiesDocs.write_markdown(
            scope.report_dir / "generate-report.md",
            [
                "# Docs Generate Report",
                "",
                f"Scope: {scope.name}",
                f"Generated files: {generated}",
            ],
        )
        return m.Infra.DocsPhaseReport(
            phase="generate",
            scope=scope.name,
            changed_files=changed,
            generated=generated,
            applied=apply,
            source="code-docstring-ssot",
            items=[
                m.Infra.DocsPhaseItemModel(
                    phase="generate", path=item.path, written=item.written
                )
                for item in files
            ],
            result=(
                c.Infra.ResultStatus.OK
                if apply or changed == 0
                else c.Infra.ResultStatus.FAIL
            ),
            reason=f"changes:{changed}",
            passed=apply or changed == 0,
        )

    @staticmethod
    def docs_project_guide_content(
        content: str, project_name: str, guide_name: str
    ) -> str:
        """Return guide content normalized for project-local publication."""
        lines = content.splitlines()
        title = Path(guide_name).stem.replace("_", " ").replace("-", " ").strip()
        body_lines = lines
        for index, line in enumerate(lines):
            match = c.Infra.HEADING_RE.match(line)
            if match is None:
                continue
            title = match.group(1).strip() or title
            body_lines = lines[index + 1 :]
            break
        body = "\n".join(body_lines).lstrip()
        heading = f"# {project_name} - {title}"
        return f"{heading}\n\n{body}".rstrip() + "\n"

    @staticmethod
    def docs_sanitize_internal_anchor_links(content: str) -> str:
        """Replace local markdown links with plain text while preserving externals."""

        def sanitize_link(match: re.Match[str]) -> str:
            target = match.group(2)
            return (
                match.group(0)
                if target.startswith(("http://", "https://", "#", "mailto:"))
                else match.group(1)
            )

        pattern: re.Pattern[str] = c.Infra.MARKDOWN_LINK_RE
        return re.sub(pattern, sanitize_link, content)


__all__: list[str] = ["FlextInfraUtilitiesDocsGenerate"]
