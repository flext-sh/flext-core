"""Release protocol dispatch: plan, version, and tag phases."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, config, m, t, u
from flext_infra.docs.generator import FlextInfraDocGenerator

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_infra import p


class FlextInfraReleaseOrchestratorDispatchMixin:
    """Core release flow and the phases that decide and record a release."""

    if TYPE_CHECKING:
        phase: c.Infra.ReleasePhase
        index: bool
        pr_title: str

        @property
        def root(self) -> Path: ...

        @property
        def logger(self) -> p.Logger: ...

        @property
        def project_names(self) -> t.StrSequence | None: ...

        @property
        def effective_dry_run(self) -> bool: ...

        def phase_build(
            self, ctx: m.Infra.ReleasePhaseDispatchConfig
        ) -> p.Result[bool]: ...

        def phase_publish(
            self, ctx: m.Infra.ReleasePhaseDispatchConfig
        ) -> p.Result[bool]: ...

    def execute(self) -> p.Result[bool]:
        """Resolve the declared version once and dispatch the selected phase."""
        current = u.Infra.current_workspace_version(self.root)
        if current.failure:
            return r[bool].fail(current.error or "project version unresolved")
        ctx = m.Infra.ReleasePhaseDispatchConfig(
            phase=self.phase,
            repository_root=self.root,
            version=current.value,
            tag=c.Infra.TAG_FORMAT.format(version=current.value),
            project_names=self.project_names or (),
            dry_run=self.effective_dry_run,
            index=self.index,
            pr_title=self.pr_title,
        )
        self.logger.info(
            "release_phase_started", phase=str(ctx.phase), current=ctx.version
        )
        handlers: dict[
            c.Infra.ReleasePhase,
            Callable[[m.Infra.ReleasePhaseDispatchConfig], p.Result[bool]],
        ] = {
            c.Infra.ReleasePhase.PLAN: lambda cfg: self.phase_plan(cfg).map(
                lambda _plan: True
            ),
            c.Infra.ReleasePhase.VERSION: self.phase_version,
            c.Infra.ReleasePhase.TAG: self.phase_tag,
            c.Infra.ReleasePhase.BUILD: self.phase_build,
            c.Infra.ReleasePhase.PUBLISH: self.phase_publish,
        }
        return handlers[ctx.phase](ctx)

    # ------------------------------------------------------------------ plan

    def phase_plan(
        self, ctx: m.Infra.ReleasePhaseDispatchConfig
    ) -> p.Result[m.Infra.ReleasePlan]:
        """Derive the next version and prove no version changed outside the protocol."""
        root = ctx.repository_root
        if ctx.pr_title and not c.Infra.CONVENTIONAL_SUBJECT_RE.match(ctx.pr_title):
            return r[m.Infra.ReleasePlan].fail(
                "pull-request title must follow Conventional Commits "
                f"(type(scope)!: description): {ctx.pr_title!r}"
            )
        guard = self._guard_version_change(root, ctx.version)
        if guard.failure:
            return r[m.Infra.ReleasePlan].fail(guard.error or "version guard failed")
        latest = self._latest_tag(root)
        if latest.failure:
            return r[m.Infra.ReleasePlan].fail(latest.error or "tag listing failed")
        plan = self._derive_plan(root, ctx.version, latest.value)
        if plan.failure:
            return plan
        report_dir = u.Cli.resolve_report_dir(root, c.Infra.PROJECT, c.Infra.RK_RELEASE)
        written = u.Cli.json_write(
            report_dir / c.Infra.RELEASE_PLAN_FILENAME,
            plan.value.model_dump(mode="json", exclude_computed_fields=True),
            m.Cli.JsonWriteOptions(sort_keys=True),
        )
        if written.failure:
            return r[m.Infra.ReleasePlan].fail(written.error or "write plan failed")
        self.logger.info(
            "release_plan",
            current=plan.value.current,
            next=plan.value.next,
            bump=str(plan.value.bump),
            releasable=plan.value.releasable,
        )
        return plan

    def _derive_plan(
        self, root: Path, current: str, latest_tag: str
    ) -> p.Result[m.Infra.ReleasePlan]:
        """Apply the protocol's decision rules to the repository state."""
        # Why: the merged release commit is not always HEAD (CI plans on the
        # pull request's synthetic merge commit), so it is looked up in the
        # whole history since the last tag.
        history = self._subjects(root, latest_tag, merges_only=False)
        if history.failure:
            return r[m.Infra.ReleasePlan].fail(history.error or "history log failed")
        current_tag = c.Infra.TAG_FORMAT.format(version=current)
        if latest_tag != current_tag and any(
            u.Infra.is_release_subject(subject, current) for subject in history.value
        ):
            # The release commit is merged and awaits its tag: nothing to bump.
            return r[m.Infra.ReleasePlan].ok(
                m.Infra.ReleasePlan(
                    current=current,
                    next=current,
                    bump=c.Infra.VersionBump.NONE,
                    previous_tag=latest_tag or None,
                )
            )
        final = u.Infra.finalize_version(current)
        if final.failure:
            return r[m.Infra.ReleasePlan].fail(final.error or "invalid version")
        declared_ahead = self._declared_ahead_of_tag(final.value, latest_tag)
        if declared_ahead.failure:
            return r[m.Infra.ReleasePlan].fail(declared_ahead.error or "tag unparsable")
        if not latest_tag or final.value != current or declared_ahead.value:
            # A first release ships the declared version as is; a declared
            # pre-release ships its base; a declared version beyond the last
            # tag ships as declared. Each of those releases was decided when
            # the version was written, so the titles merged since the tag are
            # not consulted. Titles decide only once the declared version has
            # been tagged.
            return r[m.Infra.ReleasePlan].ok(
                m.Infra.ReleasePlan(
                    current=current,
                    next=final.value,
                    bump=c.Infra.VersionBump.NONE,
                    previous_tag=latest_tag or None,
                    declared=True,
                )
            )
        merges = self._subjects(root, latest_tag, merges_only=True)
        if merges.failure:
            return r[m.Infra.ReleasePlan].fail(merges.error or "merge log failed")
        bump = u.Infra.plan_bump(merges.value, config.Infra.release.bump_types)
        if bump.failure:
            return r[m.Infra.ReleasePlan].fail(bump.error or "bump derivation failed")
        next_version = u.Infra.bump_version(current, bump.value)
        if next_version.failure:
            return r[m.Infra.ReleasePlan].fail(next_version.error or "bump failed")
        return r[m.Infra.ReleasePlan].ok(
            m.Infra.ReleasePlan(
                current=current,
                next=next_version.value,
                bump=bump.value,
                previous_tag=latest_tag,
                merges=tuple(merges.value),
            )
        )

    @staticmethod
    def _declared_ahead_of_tag(version: str, latest_tag: str) -> p.Result[bool]:
        """Whether the declared (final) version is newer than the last tag's.

        The comparison keeps pre-release segments: a repository at ``0.12.0``
        whose last tag is ``v0.12.0rc2`` is ahead of it.
        """
        if not latest_tag:
            return r[bool].ok(True)
        tag_prefix = c.Infra.TAG_FORMAT.format(version="")
        return u.Infra.version_is_newer(version, latest_tag.removeprefix(tag_prefix))

    def _guard_version_change(self, root: Path, version: str) -> p.Result[bool]:
        """Reject a pyproject version that differs from the integration base.

        The only legitimate diff is the protocol's own release commit, whose
        subject names exactly the version it introduces.
        """
        branch = self._integration_branch(root)
        if branch.failure:
            return r[bool].fail(branch.error or "integration branch unresolved")
        base = u.Cli.capture(
            [
                c.Infra.GIT,
                "merge-base",
                f"{c.Infra.GIT_ORIGIN}/{branch.value}",
                c.Infra.GIT_HEAD,
            ],
            cwd=root,
        )
        if base.failure:
            return r[bool].fail(base.error or "merge-base failed")
        base_oid = base.value.strip()
        head_oid = u.Cli.capture([c.Infra.GIT, "rev-parse", c.Infra.GIT_HEAD], cwd=root)
        if head_oid.failure:
            return r[bool].fail(head_oid.error or "rev-parse HEAD failed")
        if base_oid == head_oid.value.strip():
            return r[bool].ok(True)
        base_content = u.Cli.capture(
            [c.Infra.GIT, "show", f"{base_oid}:{c.Infra.PYPROJECT_FILENAME}"], cwd=root
        )
        if base_content.failure:
            return r[bool].fail(base_content.error or "base pyproject unreadable")
        base_match = c.Infra.VERSION_RE.search(base_content.value)
        base_version = base_match.group(1) if base_match else ""
        if base_version == version:
            return r[bool].ok(True)
        # Why: CI checks out the pull request's synthetic merge commit, and an
        # open release lane may carry integration merges above its release
        # commit; the protocol's commit is therefore looked up in the whole
        # base..HEAD range, not only at HEAD.
        subjects = self._subjects(root, base_oid, merges_only=False)
        if subjects.failure:
            return r[bool].fail(subjects.error or "git log failed")
        if any(
            u.Infra.is_release_subject(subject, version) for subject in subjects.value
        ):
            return r[bool].ok(True)
        release_subject = c.Infra.RELEASE_COMMIT_SUBJECT.format(version=version)
        return r[bool].fail(
            f"{c.Infra.PYPROJECT_FILENAME} version changed outside the release "
            f"protocol: {base_version} -> {version} (HEAD {head_oid.value.strip()[:12]} "
            f"carries no {release_subject!r}); run `make release WHAT=version APPLY=Y` "
            "instead"
        )

    # --------------------------------------------------------------- version

    def phase_version(self, ctx: m.Infra.ReleasePhaseDispatchConfig) -> p.Result[bool]:
        """Open or update the release pull request for the planned version."""
        root = ctx.repository_root
        plan = self.phase_plan(ctx)
        if plan.failure:
            return r[bool].fail(plan.error or "release plan failed")
        if not plan.value.releasable:
            self.logger.info("release_version_none", current=ctx.version)
            return r[bool].ok(True)
        if ctx.dry_run:
            return r[bool].ok(True)
        preflight = self._preflight_version(root, plan.value)
        if preflight.failure:
            return preflight
        branch = self._integration_branch(root)
        if branch.failure:
            return r[bool].fail(branch.error or "integration branch unresolved")
        for step in (
            lambda: self._switch_release_branch(root, branch.value),
            lambda: self._stamp_release(ctx, plan.value),
            lambda: self._commit_release(root, plan.value),
            lambda: self._publish_release_branch(root, branch.value, plan.value),
        ):
            outcome = step()
            if outcome.failure:
                return outcome
        self.logger.info("release_version_pull_request", version=plan.value.next)
        return r[bool].ok(True)

    def _preflight_version(
        self, root: Path, plan: m.Infra.ReleasePlan
    ) -> p.Result[bool]:
        """Require a clean checkout on the integration branch and a free tag."""
        status = u.Infra.git_status(m.Infra.GitStatusRequest(repo_root=root))
        if status.failure:
            return r[bool].fail(status.error or "git status failed")
        if status.value.dirty:
            return r[bool].fail(
                f"release version requires a clean checkout: {root}\n"
                f"{status.value.porcelain}"
            )
        branch = self._integration_branch(root)
        if branch.failure:
            return r[bool].fail(branch.error or "integration branch unresolved")
        current = u.Infra.git_current_branch(m.Infra.GitRepoRequest(repo_root=root))
        if current.failure:
            return r[bool].fail(current.error or "current branch unresolved")
        if current.value.text != branch.value:
            return r[bool].fail(
                f"release version runs on {branch.value}, not {current.value.text}"
            )
        exists = u.Cli.capture([c.Infra.GIT, "tag", "-l", plan.tag], cwd=root)
        if exists.failure:
            return r[bool].fail(exists.error or "tag check failed")
        if exists.value.strip():
            return r[bool].fail(f"release tag already exists: {plan.tag}")
        return r[bool].ok(True)

    def _switch_release_branch(self, root: Path, integration: str) -> p.Result[bool]:
        """Continue the open release lane when it exists, else start it from HEAD."""
        remote_ref = f"refs/remotes/{c.Infra.GIT_ORIGIN}/{c.Infra.RELEASE_BRANCH}"
        # Why: a rerun (CI retry, or a local run after a first attempt) must
        # continue the same lane, whether it already exists locally or only
        # on the remote, and never fail on "branch already exists".
        local = u.Cli.capture(
            [
                c.Infra.GIT,
                "rev-parse",
                "--verify",
                "--quiet",
                f"refs/heads/{c.Infra.RELEASE_BRANCH}",
            ],
            cwd=root,
        )
        fetch = u.Cli.run_checked(
            [c.Infra.GIT, "fetch", c.Infra.GIT_ORIGIN, c.Infra.RELEASE_BRANCH], cwd=root
        )
        if local.success:
            switched = u.Cli.run_checked(
                [c.Infra.GIT, "switch", c.Infra.RELEASE_BRANCH], cwd=root
            )
        elif fetch.success:
            switched = u.Cli.run_checked(
                [c.Infra.GIT, "switch", "--create", c.Infra.RELEASE_BRANCH, remote_ref],
                cwd=root,
            )
        else:
            return u.Cli.run_checked(
                [c.Infra.GIT, "switch", "--create", c.Infra.RELEASE_BRANCH], cwd=root
            )
        if switched.failure:
            return switched
        return u.Infra.git_merge_no_edit(
            m.Infra.GitCommitishRequest(repo_root=root, commitish=integration)
        ).map(lambda _report: True)

    def _stamp_release(
        self, ctx: m.Infra.ReleasePhaseDispatchConfig, plan: m.Infra.ReleasePlan
    ) -> p.Result[bool]:
        """Write the version SSOT, the release notes, and the changelog."""
        root = ctx.repository_root
        stamped = u.Infra.replace_project_version(root, plan.next)
        if stamped.failure:
            return stamped
        # Why: the lock records the project's own version, so the stamp
        # refreshes it the way `make deps WHAT=lock APPLY=Y` does; otherwise
        # `make deps` (uv lock --check) is red on the release lane.
        locked = u.Cli.run_checked(
            [c.Infra.UV, "lock", "--project", str(root)], cwd=root
        )
        if locked.failure:
            return locked
        notes_path = (
            u.Cli.resolve_report_dir(root, c.Infra.PROJECT, c.Infra.RK_RELEASE)
            / plan.tag
            / c.Infra.RELEASE_NOTES_FILENAME
        )
        projects = u.Infra.resolve_projects(root, ctx.project_names)
        if projects.failure:
            return r[bool].fail(projects.error or "release project resolution failed")
        notes = u.Infra.generate_notes(
            plan.next,
            plan.tag,
            projects.value,
            "\n".join(f"- {subject}" for subject in plan.merges),
            notes_path,
        )
        if notes.failure:
            return notes
        changelog = u.Infra.update_changelog(root, plan.next, plan.tag, notes_path)
        if changelog.failure:
            return changelog
        # Why: README, docs/index and the API overview render the version, and
        # the docs generator owns them; the stamp regenerates its projections
        # so `make gen WHAT=check` stays a fixed point on the release lane.
        return FlextInfraDocGenerator(
            workspace=root, projects=ctx.project_names or None, apply=True
        ).execute()

    def _commit_release(self, root: Path, plan: m.Infra.ReleasePlan) -> p.Result[bool]:
        """Commit the stamped SSOT and every projection regenerated from it.

        Preflight proved the checkout clean, so every path the status now
        lists was produced by the stamp; those exact paths are staged. A
        rerun stamps the bytes the lane already carries and commits nothing.
        """
        status = u.Infra.git_status(m.Infra.GitStatusRequest(repo_root=root))
        if status.failure:
            return r[bool].fail(status.error or "git status failed")
        # The status code and the path are whitespace-separated; the first
        # line arrives without its leading status padding.
        produced = tuple(
            line.split(maxsplit=1)[1]
            for line in status.value.porcelain.splitlines()
            if line.strip()
        )
        if not produced:
            self.logger.info("release_version_unchanged", version=plan.next)
            return r[bool].ok(True)
        staged = u.Infra.git_add_paths(
            m.Infra.GitPathsRequest(repo_root=root, paths=produced)
        )
        if staged.failure:
            return r[bool].fail(staged.error or "git add failed")
        committed = u.Infra.git_commit(
            m.Infra.GitCommitRequest(
                repo_root=root,
                message=c.Infra.RELEASE_COMMIT_SUBJECT.format(version=plan.next),
            )
        )
        if committed.failure:
            return r[bool].fail(committed.error or "git commit failed")
        return r[bool].ok(True)

    def _publish_release_branch(
        self, root: Path, integration: str, plan: m.Infra.ReleasePlan
    ) -> p.Result[bool]:
        """Push the release lane and open or update its pull request."""
        pushed = u.Infra.git_push_upstream(
            m.Infra.GitPushRequest(repo_root=root, branch=c.Infra.RELEASE_BRANCH)
        )
        if pushed.failure:
            return r[bool].fail(pushed.error or "git push failed")
        title = c.Infra.RELEASE_COMMIT_SUBJECT.format(version=plan.next)
        body = (
            u.Cli.resolve_report_dir(root, c.Infra.PROJECT, c.Infra.RK_RELEASE)
            / plan.tag
            / c.Infra.RELEASE_NOTES_FILENAME
        )
        exists = u.Cli.capture(
            [c.Infra.GH, "pr", "view", c.Infra.RELEASE_BRANCH, "--json", "number"],
            cwd=root,
        )
        command = (
            [c.Infra.GH, "pr", "edit", c.Infra.RELEASE_BRANCH]
            if exists.success
            else [
                c.Infra.GH,
                "pr",
                "create",
                "--base",
                integration,
                "--head",
                c.Infra.RELEASE_BRANCH,
            ]
        )
        return u.Cli.run_checked(
            [*command, "--title", title, "--body-file", str(body)], cwd=root
        )

    # ------------------------------------------------------------------- tag

    def phase_tag(self, ctx: m.Infra.ReleasePhaseDispatchConfig) -> p.Result[bool]:
        """Tag the merged release commit; idempotent when the tag already points here."""
        root = ctx.repository_root
        head = self._head_subject(root)
        if head.failure:
            return r[bool].fail(head.error or "HEAD subject failed")
        expected = c.Infra.RELEASE_COMMIT_SUBJECT.format(version=ctx.version)
        if not u.Infra.is_release_subject(head.value, ctx.version):
            return r[bool].fail(
                f"release tag requires HEAD to be the release commit {expected!r}, "
                f"found {head.value!r}"
            )
        if ctx.dry_run:
            return r[bool].ok(True)
        created = self._create_tag(root, ctx.tag)
        if created.failure:
            return created
        return u.Cli.run_checked(
            [c.Infra.GIT, "push", c.Infra.GIT_ORIGIN, ctx.tag], cwd=root
        )

    def _create_tag(self, repository_root: Path, tag: str) -> p.Result[bool]:
        """Create the annotated tag at HEAD, or accept it when it already points there."""
        existing = u.Cli.capture(
            [c.Infra.GIT, "rev-list", "-n", "1", tag], cwd=repository_root
        )
        if existing.success and existing.value.strip():
            head = u.Cli.capture(
                [c.Infra.GIT, "rev-parse", c.Infra.GIT_HEAD], cwd=repository_root
            )
            if head.failure:
                return r[bool].fail(head.error or "rev-parse HEAD failed")
            if existing.value.strip() != head.value.strip():
                return r[bool].fail(f"release tag {tag} already points elsewhere")
            return r[bool].ok(True)
        return u.Cli.run_checked(
            [c.Infra.GIT, "tag", "-a", tag, "-m", f"release: {tag}"],
            cwd=repository_root,
        )

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _integration_branch(root: Path) -> p.Result[str]:
        """Resolve the published integration branch this repository releases from."""
        return u.Infra.repository_baseline_branch(
            root,
            preference=tuple(
                config.Infra.codegen.branch_policy.integration_branch_preference
            ),
        )

    @staticmethod
    def _head_subject(root: Path) -> p.Result[str]:
        """Return the HEAD commit subject."""
        return u.Cli.capture(
            [c.Infra.GIT, "log", "-1", "--format=%s", c.Infra.GIT_HEAD], cwd=root
        ).map(str.strip)

    @staticmethod
    def _latest_tag(root: Path) -> p.Result[str]:
        """Return the highest release tag, or an empty string before the first release.

        The listing is deliberately unsorted: Git's version collation is not a
        PEP 440 ordering and ranks ``v0.12.0rc2`` above ``v0.12.0``. The typed
        versioning owner performs the ordering instead.
        """
        tags = u.Cli.capture(
            [c.Infra.GIT, "tag", "--list", c.Infra.TAG_FORMAT.format(version="*")],
            cwd=root,
        )
        if tags.failure:
            return r[str].fail(tags.error or "release tag listing failed")
        return u.Infra.latest_release_tag(tags.value.splitlines())

    @staticmethod
    def _subjects(
        root: Path, since: str, *, merges_only: bool
    ) -> p.Result[t.StrSequence]:
        """Return commit subjects reachable from HEAD since ``since`` (all when empty).

        Merge commits carry the pull-request titles the bump is derived from;
        the full history is what the release commit is looked up in. GitHub's
        default merge subject contains only the PR number and branch, so its
        first body line is the authoritative PR title. Preserve the default
        subject when that line is absent so release planning fails loud.
        """
        log = u.Cli.capture(
            [
                c.Infra.GIT,
                "log",
                "--format=%s%x1f%b%x1e",
                *(("--merges",) if merges_only else ()),
                f"{since}..{c.Infra.GIT_HEAD}",
            ],
            cwd=root,
        )
        if log.failure:
            return r[t.StrSequence].fail(log.error or "merge log failed")
        titles: list[str] = []
        for record in log.value.split("\x1e"):
            if not record.strip():
                continue
            subject, _, body = record.strip().partition("\x1f")
            if merges_only and c.Infra.PULL_REQUEST_MERGE_SUBJECT_RE.match(subject):
                title = next(
                    (line.strip() for line in body.splitlines() if line.strip()), ""
                )
                titles.append(title or subject)
            else:
                titles.append(subject)
        return r[t.StrSequence].ok(tuple(titles))


__all__: list[str] = ["FlextInfraReleaseOrchestratorDispatchMixin"]
