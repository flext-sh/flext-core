"""Fix helpers for docs services."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from flext_cli import u
from flext_infra._utilities._docs_github_links import FlextInfraUtilitiesDocsGithubLinks
from flext_infra._utilities.docs import FlextInfraUtilitiesDocs
from flext_infra._utilities.docs_contract import FlextInfraUtilitiesDocsContract
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.typings import t

if TYPE_CHECKING:
    import re
    from pathlib import Path


class FlextInfraUtilitiesDocsFix:
    """Reusable fix helpers exposed through ``u.Infra``."""

    @staticmethod
    def docs_maybe_fix_link(md_file: Path, raw_link: str) -> str | None:
        """Return a corrected link target when a simple fix is possible."""
        if raw_link.startswith(("http://", "https://")):
            return FlextInfraUtilitiesDocsGithubLinks.docs_rewrite_github_url(raw_link)
        result: str | None = None
        if not raw_link.startswith(("mailto:", "tel:", "#")):
            base = raw_link.split("#", maxsplit=1)[0]
            if (
                base
                and not (md_file.parent / base).exists()
                and not base.endswith(".md")
            ):
                md_candidate = md_file.parent / f"{base}.md"
                if md_candidate.exists():
                    result = f"{base}.md{raw_link[len(base) :]}"
        return result

    @staticmethod
    def docs_fix_python_codeblocks(
        scope: m.Infra.DocScope, *, apply: bool
    ) -> t.SequenceOf[m.Infra.GeneratedFile]:
        """Auto-fix ``python`` fenced code blocks using ``ruff check --fix``.

        Only fixes issues that ``ruff`` can resolve automatically; blocks that
        still contain unfixable diagnostics are left untouched so the audit
        gate reports them.
        """
        changed: t.MutableSequenceOf[m.Infra.GeneratedFile] = []
        for md_file in FlextInfraUtilitiesDocs.iter_scope_markdown_files(scope):
            original = md_file.read_text(
                encoding=c.Cli.ENCODING_DEFAULT, errors=c.Infra.IGNORE
            )

            def _replace_fence(
                match: re.Match[str], source_file: Path = md_file
            ) -> str:
                body = match.group("body")
                rel = source_file.relative_to(scope.path).as_posix()
                # flext-o6h5 (agent: kimi) — ruff via running interpreter (venv SSOT);
                # bare "ruff" breaks when .venv/bin is not on PATH (CI docs fix).
                outcome = u.Cli.run_raw(
                    [
                        sys.executable,
                        "-m",
                        c.Infra.RUFF,
                        c.Infra.VERB_CHECK,
                        "--fix",
                        "--extend-ignore",
                        ",".join(c.Infra.PYTHON_FENCE_RUFF_EXTEND_IGNORE),
                        "--stdin-filename",
                        f"{rel}#block.py",
                        "-",
                    ],
                    input_data=body.encode(),
                )
                if outcome.failure:
                    return match.group(0)
                fixed_body = outcome.value.stdout
                if fixed_body == body:
                    return match.group(0)
                # A closing fence only closes the block when it starts its own
                # line; ruff may return a body without the trailing newline, so
                # reassembling blindly welds the fence onto the last code line
                # and the block swallows every heading that follows it.
                closed_body = (
                    fixed_body if fixed_body.endswith("\n") else f"{fixed_body}\n"
                )
                return f"{match.group('open')}{closed_body}```"

            sanitized = c.Infra.PYTHON_FENCE_FIX_RE.sub(_replace_fence, original)
            if sanitized == original:
                continue
            changed.append(
                FlextInfraUtilitiesDocsContract.docs_write_if_needed(
                    md_file, sanitized, apply=apply
                )
            )
        return changed

    @staticmethod
    def docs_process_markdown_file(
        md_file: Path, *, apply: bool
    ) -> m.Infra.DocsPhaseItemModel:
        """Fix one markdown file and return the phase item summary."""
        original = md_file.read_text(
            encoding=c.Cli.ENCODING_DEFAULT, errors=c.Infra.IGNORE
        )
        link_count = 0

        def replace_link(match: t.Infra.RegexMatch) -> str:
            """Replace link."""
            nonlocal link_count
            text, link = match.groups()
            fixed = FlextInfraUtilitiesDocsFix.docs_maybe_fix_link(md_file, link)
            if fixed is None:
                original_match: str = match.group(0)
                return original_match
            link_count += 1
            return f"[{text}]({fixed})"

        updated = c.Infra.MARKDOWN_LINK_RE.sub(replace_link, original)
        updated, toc_changed = FlextInfraUtilitiesDocs.update_toc(updated)
        if apply and (link_count > 0 or toc_changed > 0) and updated != original:
            _ = md_file.write_text(updated, encoding=c.Cli.ENCODING_DEFAULT)
        return m.Infra.DocsPhaseItemModel(
            phase="fix", file=md_file.as_posix(), links=link_count, toc=toc_changed
        )

    @staticmethod
    def docs_write_fix_reports(
        scope: m.Infra.DocScope,
        *,
        items: t.SequenceOf[m.Infra.DocsPhaseItemModel],
        apply: bool,
    ) -> None:
        """Persist the standard fix summary and markdown report."""
        changes_payload: t.JsonList = [
            {c.Infra.RK_FILE: item.file, "links": item.links, "toc": item.toc}
            for item in items
        ]
        summary_payload = t.Cli.JSON_MAPPING_ADAPTER.validate_python({
            c.Infra.RK_SUMMARY: {
                c.Infra.RK_SCOPE: scope.name,
                "changed_files": len(items),
                "apply": apply,
            },
            "changes": changes_payload,
        })
        _ = u.Cli.json_write(scope.report_dir / "fix-summary.json", summary_payload)
        _ = FlextInfraUtilitiesDocs.write_markdown(
            scope.report_dir / "fix-report.md",
            [
                "# Docs Fix Report",
                "",
                f"Scope: {scope.name}",
                f"Apply: {int(apply)}",
                f"Changed files: {len(items)}",
                "",
                "| file | link_fixes | toc_updates |",
                "|---|---:|---:|",
                *[f"| {item.file} | {item.links} | {item.toc} |" for item in items],
            ],
        )


__all__: list[str] = ["FlextInfraUtilitiesDocsFix"]
