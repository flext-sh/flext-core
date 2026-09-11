"""Pure planning for the project-layout engine (flext-0wuz, epic flext-hzox).

Every classification derives from ``config.Infra.codegen.layout`` — this mixin
reads the declarative SSOT and produces typed findings; it never writes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path

from flext_infra import c, config, m, t, u


class FlextInfraCodegenLayoutPlanMixin:
    """Build the per-project layout plan from the declarative layout SSOT."""

    @property
    def _layout_spec(self) -> m.Infra.LayoutSpec:
        """Layout SSOT loaded once through the validated config singleton."""
        return config.Infra.codegen.layout

    def plan_project(self, project_dir: Path) -> m.Infra.LayoutProjectReport:
        """Classify every root entry of one project without writing anything."""
        spec = self._layout_spec
        project_name = project_dir.name
        override = self._resolve_override(spec, project_name)
        allowed = self._allowed_root_names(spec, project_name, override)
        override_roots = self._override_root_names(override)
        findings: list[m.Infra.LayoutFinding] = []
        for entry in sorted(project_dir.iterdir()):
            name = entry.name
            if spec.allow_hidden and name.startswith("."):
                continue
            if self._is_ignored_root(spec, override, name):
                continue
            if name in allowed or name in override_roots:
                continue
            findings.append(
                self._classify_root_entry(spec, override, project_name, entry)
            )
        if override is not None:
            findings.extend(self._override_move_findings(override, project_dir))
            findings.extend(
                self._override_empty_dir_findings(spec, override, project_dir)
            )
        findings.extend(self._gitignore_findings(spec, override, project_dir))
        return m.Infra.LayoutProjectReport(
            project=project_name, findings=tuple(findings)
        )

    @staticmethod
    def _resolve_override(
        spec: m.Infra.LayoutSpec, project_name: str
    ) -> m.Infra.LayoutProjectOverrideSpec | None:
        """Resolve override by directory name or logical name without leading dot."""
        override = spec.project_overrides.get(project_name)
        if override is not None:
            return override
        logical = project_name.lstrip(".")
        if logical != project_name:
            return spec.project_overrides.get(logical)
        return None

    def _allowed_root_names(
        self,
        spec: m.Infra.LayoutSpec,
        project_name: str,
        override: m.Infra.LayoutProjectOverrideSpec | None,
    ) -> frozenset[str]:
        """Canonical root names for one project, profile extras included."""
        allowed = {
            *spec.canonical_root_files,
            *spec.canonical_root_dotfiles,
            *spec.canonical_root_dirs,
            *spec.special_root_dirs,
            *spec.reference_root_dirs,
        }
        for profile, patterns in spec.profile_project_patterns.items():
            if any(fnmatchcase(project_name, pattern) for pattern in patterns):
                allowed.update(spec.profile_extra_root_files.get(profile, ()))
        if override is not None:
            allowed.update(override.keep_root_files)
        return frozenset(allowed)

    @staticmethod
    def _is_ignored_root(
        spec: m.Infra.LayoutSpec,
        override: m.Infra.LayoutProjectOverrideSpec | None,
        name: str,
    ) -> bool:
        """Whether a root entry is skipped by specials or project ignore globs."""
        if name in spec.special_root_dirs:
            return True
        if override is None:
            return False
        return any(fnmatchcase(name, glob) for glob in override.ignore_globs)

    @staticmethod
    def _override_root_names(
        override: m.Infra.LayoutProjectOverrideSpec | None,
    ) -> frozenset[str]:
        """Root names owned by per-project override rules (never generic rules)."""
        if override is None:
            return frozenset()
        return frozenset({
            *(Path(move.source).parts[0] for move in override.moves),
            *override.archive_empty_dirs,
        })

    def _classify_root_entry(
        self,
        spec: m.Infra.LayoutSpec,
        override: m.Infra.LayoutProjectOverrideSpec | None,
        project_name: str,
        entry: Path,
    ) -> m.Infra.LayoutFinding:
        """Classify one non-canonical root entry into move/archive/review."""
        name = entry.name
        if entry.is_dir() and name in spec.move_docs_dirs:
            return self._finding("move", name, f"{spec.docs_target}/{name}")
        if entry.is_file() and name in spec.move_docs_files:
            target = f"{spec.docs_target}/{name}"
            if (entry.parent / target).exists():
                return self._finding(
                    "archive",
                    name,
                    f"{spec.archive_root}/{project_name}/{name}",
                    message=(
                        f"archive {name} -> {spec.archive_root}/{project_name}/{name} "
                        f"(canonical docs/{name} kept)"
                    ),
                )
            return self._finding("move", name, target)
        if entry.is_file() and name in spec.move_example_files:
            return self._finding("move", name, f"{spec.examples_target}/{name}")
        if entry.is_file() and any(
            fnmatchcase(name, glob) for glob in spec.move_diagram_globs
        ):
            return self._finding("move", name, f"{spec.diagrams_target}/{name}")
        if (
            name in spec.archive_names
            or (override is not None and name in override.archive_names)
            or any(fnmatchcase(name, glob) for glob in spec.archive_globs)
        ):
            return self._finding(
                "archive", name, f"{spec.archive_root}/{project_name}/{name}"
            )
        return self._finding("review", name)

    def _override_move_findings(
        self, override: m.Infra.LayoutProjectOverrideSpec, project_dir: Path
    ) -> t.SequenceOf[m.Infra.LayoutFinding]:
        """Explicit per-project moves whose nested source still exists."""
        return tuple(
            self._finding("move", move.source, move.target)
            for move in override.moves
            if (project_dir / move.source).exists()
        )

    def _override_empty_dir_findings(
        self,
        spec: m.Infra.LayoutSpec,
        override: m.Infra.LayoutProjectOverrideSpec,
        project_dir: Path,
    ) -> t.SequenceOf[m.Infra.LayoutFinding]:
        """Override directories archived once override moves have emptied them."""
        findings: list[m.Infra.LayoutFinding] = []
        move_sources = {move.source for move in override.moves}
        for name in override.archive_empty_dirs:
            path = project_dir / name
            if not path.is_dir():
                continue
            target = f"{spec.archive_root}/{project_dir.name}/{name}"
            if not any(path.iterdir()):
                findings.append(self._finding("archive", name, target))
                continue
            remaining = {
                entry.relative_to(project_dir).as_posix()
                for entry in path.rglob("*")
                if entry.is_file()
            }
            if remaining and remaining <= move_sources:
                findings.append(self._finding("archive", name, target))
                continue
            findings.append(self._finding("review", name))
        return tuple(findings)

    def _gitignore_findings(
        self,
        spec: m.Infra.LayoutSpec,
        override: m.Infra.LayoutProjectOverrideSpec | None,
        project_dir: Path,
    ) -> t.SequenceOf[m.Infra.LayoutFinding]:
        """Missing ``.gitignore`` patterns required by the layout SSOT."""
        required = [f"{spec.archive_root}/"]
        if override is not None:
            required.extend(override.gitignore_additions)
        current = ""
        gitignore_path = project_dir / c.Infra.GITIGNORE
        if gitignore_path.is_file():
            read = u.Cli.files_read_text(gitignore_path)
            if read.success:
                current = read.value
        covered = {line.strip() for line in current.splitlines()}
        findings: list[m.Infra.LayoutFinding] = []
        for pattern in required:
            if pattern in covered or pattern.rstrip("/") in covered:
                continue
            findings.append(self._finding("gitignore", c.Infra.GITIGNORE, pattern))
        return tuple(findings)

    @staticmethod
    def _finding(
        rule: t.Infra.LayoutRule,
        path: str,
        target: str = "",
        *,
        message: str | None = None,
    ) -> m.Infra.LayoutFinding:
        """Build one typed finding with a canonical message."""
        messages: t.MappingKV[t.Infra.LayoutRule, str] = {
            "move": f"move {path} -> {target}",
            "archive": f"archive {path} -> {target}",
            "gitignore": f"add .gitignore entry: {target}",
            "review": f"non-canonical root entry requires review: {path}",
        }
        return m.Infra.LayoutFinding(
            rule=rule,
            path=path,
            target=target,
            message=message if message is not None else messages[rule],
        )


__all__: list[str] = ["FlextInfraCodegenLayoutPlanMixin"]
