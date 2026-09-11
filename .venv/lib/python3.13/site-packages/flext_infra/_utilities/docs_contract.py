"""Reusable docs contract helpers exposed through ``u.Infra``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from flext_cli import u
from flext_infra._utilities.docs_scope import FlextInfraUtilitiesDocsScope
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.typings import t

if TYPE_CHECKING:
    from pathlib import Path


class FlextInfraUtilitiesDocsContract:
    """Contract helpers for docs services."""

    @staticmethod
    def _docs_body_start(lines: list[str]) -> int:
        """Return the first body line index, skipping YAML frontmatter when present."""
        if not lines or lines[0].strip() != "---":
            return 0
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                return index + 1
        return 0

    @staticmethod
    def _docs_strip_invented_toc_before_frontmatter(content: str) -> str:
        """Undo H1+TOC wrongly prepended ahead of YAML frontmatter."""
        if not content.startswith("# Documentation"):
            return content
        if c.Infra.TOC_START not in content or c.Infra.TOC_END not in content:
            return content
        toc_end = content.find(c.Infra.TOC_END)
        if toc_end < 0:
            return content
        after_toc = content[toc_end + len(c.Infra.TOC_END) :].lstrip("\n")
        if not after_toc.startswith("---"):
            return content
        return after_toc

    @staticmethod
    def docs_update_toc(content: str) -> t.StrIntPair:
        """Normalize the managed table of contents in Markdown content."""
        original = content
        content = (
            FlextInfraUtilitiesDocsContract._docs_strip_invented_toc_before_frontmatter(
                content
            )
        )
        toc = FlextInfraUtilitiesDocsContract.docs_build_toc(content)
        if c.Infra.TOC_START in content and c.Infra.TOC_END in content:
            updated = c.Infra.TOC_BLOCK_RE.sub(toc, content, count=1)
            return (updated, int(updated != original))
        lines = content.splitlines()
        body_start = FlextInfraUtilitiesDocsContract._docs_body_start(lines)
        heading_at = next(
            (
                index
                for index, line in enumerate(lines[body_start:], start=body_start)
                if line.startswith("# ")
                or (
                    line.strip()
                    and not line.lstrip().startswith("<!--")
                    and line.strip() != "---"
                )
            ),
            None,
        )
        if heading_at is not None and lines[heading_at].startswith("# "):
            insert_at = heading_at + 1
            while insert_at < len(lines) and (not lines[insert_at].strip()):
                insert_at += 1
            lines[heading_at + 1 : insert_at] = [""]
            lines.insert(heading_at + 2, toc)
            lines.insert(heading_at + 3, "")
            updated = "\n".join(lines) + ("\n" if content.endswith("\n") else "")
            return (updated, int(updated != original))
        # MD041 wants an H1 for heading-less bodies. Keep YAML frontmatter first.
        body = "\n".join(lines[body_start:]).lstrip()
        prefix = "\n".join(lines[:body_start])
        invented = f"# Documentation\n\n{toc}\n\n{body}"
        updated = f"{prefix}\n{invented}" if prefix else invented
        if content.endswith("\n") and not updated.endswith("\n"):
            updated = f"{updated}\n"
        return (updated, int(updated != original))

    @staticmethod
    def docs_anchorize(text: str) -> str:
        """Convert a heading title to a GitHub-compatible anchor slug."""
        normalized: str = u.norm_str(text, case="lower")
        alnum_only: str = c.Infra.ANCHOR_NON_ALNUM_RE.sub("", normalized)
        collapsed_whitespace: str = c.Infra.ANCHOR_WHITESPACE_RE.sub("-", alnum_only)
        slug: str = c.Infra.ANCHOR_DASH_COLLAPSE_RE.sub(
            "-", collapsed_whitespace
        ).strip("-")
        return slug

    @staticmethod
    def docs_build_toc(content: str) -> str:
        """Generate a managed TOC block from second- and third-level headings.

        Headings inside fenced code blocks are documentation samples, not
        sections: markdown renders them verbatim and emits no anchor, so a TOC
        entry pointing at them is a dead link that fails the strict docs build.
        """
        scannable: str = c.Infra.FENCED_BLOCK_RE.sub("", content)
        items: t.MutableSequenceOf[str] = []
        for level, title in c.Infra.HEADING_H2_H3_RE.findall(scannable):
            anchor = FlextInfraUtilitiesDocsContract.docs_anchorize(title)
            if anchor:
                indent = "  " if level == "###" else ""
                items.append(f"{indent}- [{title}](#{anchor})")
        if not items:
            items = ["- No sections found"]
        return f"{c.Infra.TOC_START}\n" + "\n".join(items) + f"\n{c.Infra.TOC_END}"

    @staticmethod
    def docs_workspace_contract(repository_root: Path) -> t.JsonMapping:
        """Return the root docs contract using root ``pyproject.toml`` metadata."""
        payload = FlextInfraUtilitiesDocsScope.project_payload(repository_root)
        docs_meta = FlextInfraUtilitiesDocsScope.project_docs_meta(repository_root)
        exclude_docs = FlextInfraUtilitiesDocsScope.docs_meta_list(
            repository_root, "exclude_docs"
        )
        project_meta_value = payload.get(c.Infra.PROJECT)
        project_meta: t.JsonMapping = (
            t.Infra.INFRA_MAPPING_ADAPTER.validate_python(project_meta_value)
            if isinstance(project_meta_value, Mapping)
            else t.Infra.INFRA_MAPPING_ADAPTER.validate_python({})
        )
        project_urls_value = project_meta.get("urls")
        project_urls: t.JsonMapping = (
            t.Infra.INFRA_MAPPING_ADAPTER.validate_python(project_urls_value)
            if isinstance(project_urls_value, Mapping)
            else t.Infra.INFRA_MAPPING_ADAPTER.validate_python({})
        )
        result: t.JsonMapping = t.Infra.INFRA_MAPPING_ADAPTER.validate_python({
            "name": str(project_meta.get("name", "flext")).strip() or "flext",
            "description": str(project_meta.get("description", "")).strip(),
            "version": str(project_meta.get(c.Infra.VERSION, "")).strip(),
            "site_title": str(docs_meta.get("site_title", "")).strip()
            or "FLEXT Workspace",
            "site_url": str(
                project_urls.get("Documentation")
                or project_urls.get("Homepage")
                or c.Infra.GITHUB_REPO_URL
            ).strip(),
            "repo_url": str(
                project_urls.get("Repository")
                or project_urls.get("Homepage")
                or c.Infra.GITHUB_REPO_URL
            ).strip(),
            "exclude_docs": list(exclude_docs),
        })
        return result

    @staticmethod
    def docs_write_if_needed(
        path: Path, content: str, *, apply: bool, overwrite: bool = True
    ) -> m.Infra.GeneratedFile:
        """Write generated content only when needed and allowed.

        A file that does not exist cannot drift: creating it is the apply
        run's job. Check mode flags only existing-but-stale content, so a
        fresh clone passes without pre-seeding the ephemeral output
        directory.
        """
        exists = path.exists()
        if exists and not overwrite:
            return m.Infra.GeneratedFile(
                path=path.as_posix(), changed=False, written=False
            )
        current = path.read_text(encoding=c.Cli.ENCODING_DEFAULT) if exists else ""
        normalized = (
            FlextInfraUtilitiesDocsContract.docs_update_toc(content)[0]
            if path.suffix == ".md"
            else content
        )
        drift = exists and current != normalized
        created = not exists
        if apply and (drift or created):
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text(normalized, encoding=c.Cli.ENCODING_DEFAULT)
        return m.Infra.GeneratedFile(
            path=path.as_posix(),
            changed=drift or created,
            written=apply and (drift or created),
        )


__all__: list[str] = ["FlextInfraUtilitiesDocsContract"]
