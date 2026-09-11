from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, config, m, u

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraWorktreeProvisioning:
    @staticmethod
    def _ensure_gitlink_checkout(lane: Path, member_path: Path) -> p.Result[bool]:
        reference = member_path.as_posix()
        git_marker = lane / member_path / ".git"
        if git_marker.is_symlink() or (
            git_marker.exists() and not git_marker.is_file()
        ):
            return r.fail(f"governed gitlink has an invalid .git marker: {reference}")
        if git_marker.exists():
            return r.ok(True)
        initialized = u.Infra.git_submodule_init(
            m.Infra.GitRefRequest(repo_root=lane, reference=reference)
        )
        if initialized.failure:
            return r.fail(
                initialized.error
                or f"failed to initialize governed gitlink: {reference}"
            )
        return r.ok(True)

    @staticmethod
    def _verify_gitlink_state(
        lane: Path, member_path: Path, declared_url: str, recorded_oid: str
    ) -> p.Result[bool]:
        reference = member_path.as_posix()
        identity = u.Infra.git_identity(
            m.Infra.GitRepoRequest(repo_root=lane / member_path)
        )
        if identity.failure:
            return r.fail(
                identity.error or f"failed to inspect governed gitlink: {reference}"
            )
        if identity.value.dirty:
            return r.fail(f"governed gitlink is dirty: {reference}")
        origin = identity.value.origin_remote
        if origin is None or u.Infra.git_remote_identity(
            origin
        ) != u.Infra.git_remote_identity(declared_url):
            return r.fail(f"governed gitlink identity mismatch: {reference}")
        if identity.value.head_oid != recorded_oid:
            return r.fail(
                f"governed gitlink {reference} is not at recorded oid {recorded_oid}"
            )
        return r.ok(True)

    @classmethod
    def _validate_governed_gitlink(
        cls, lane: Path, member_path: Path
    ) -> p.Result[bool]:
        reference = member_path.as_posix()
        contract = u.Infra.gitmodule_contract(
            m.Infra.GitSubmoduleContractRequest(repo_root=lane, member_path=reference)
        )
        if contract.failure:
            return r.fail(contract.error or f"invalid governed gitlink: {reference}")
        recorded = u.Infra.git_staged_gitlink_oid(
            m.Infra.GitRefRequest(repo_root=lane, reference=reference)
        )
        if recorded.failure:
            return r.fail(recorded.error or f"missing governed gitlink: {reference}")
        ensured = cls._ensure_gitlink_checkout(lane, member_path)
        if ensured.failure:
            return ensured
        return cls._verify_gitlink_state(
            lane, member_path, contract.value.url, recorded.value.oid
        )

    @classmethod
    def _prepare_governed_gitlinks(cls, lane: Path) -> p.Result[bool]:
        declared = u.Infra.git_declared_submodule_paths(lane)
        if declared.failure:
            return r.fail(declared.error or "failed to read lane gitlink declarations")
        sections = u.Infra.git_submodule_sections(
            m.Infra.GitRepoRequest(repo_root=lane)
        )
        if sections.failure:
            return r.fail(sections.error or "failed to classify lane gitlinks")
        for member_path in declared.value:
            section = sections.value.get(member_path.as_posix())
            if section is None:
                return r.fail(f"lane gitlink declaration is missing: {member_path}")
            managed = u.Infra.git_submodule_config_value(
                m.Infra.GitSubmoduleConfigRequest(
                    repo_root=lane, section=section, key="flext-managed"
                )
            )
            if managed.failure:
                return r.fail(
                    managed.error or f"failed to classify gitlink: {member_path}"
                )
            if managed.value.text.lower() != "true":
                continue
            validated = cls._validate_governed_gitlink(lane, member_path)
            if validated.failure:
                return validated
        return r.ok(True)

    @classmethod
    def setup_lane(cls, lane: Path) -> p.Result[bool]:
        gitlinks = cls._prepare_governed_gitlinks(lane)
        if gitlinks.failure:
            return gitlinks
        if not (lane / c.Infra.PYPROJECT_FILENAME).is_file():
            return r.ok(True)
        venv_name = config.Infra.tooling.tools.pyright.path_rules.venv_name
        lane_venv = lane / venv_name
        if lane_venv.is_symlink():
            try:
                lane_venv.unlink()
            except OSError as exc:
                return r.fail(f"failed to remove foreign lane environment link: {exc}")
        setup = u.Cli.run_live(
            (c.Infra.MAKE, "setup"),
            cwd=lane,
            remove_env_keys=c.Infra.ORCHESTRATOR_REMOVE_ENV_KEYS,
        )
        if setup.failure:
            return r.fail(setup.error or "make setup execution failed")
        interpreter = (
            lane_venv / "Scripts" / "python.exe"
            if os.name == "nt"
            else lane_venv / "bin" / "python"
        )
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            return r.fail(f"lane setup did not create an interpreter: {interpreter}")
        return r.ok(True)


__all__: list[str] = ["FlextInfraWorktreeProvisioning"]
