"""SSH-signed Git gate attestation owner."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from pathlib import Path
import re
from typing import TYPE_CHECKING

from git import GitCommandError

from flext_cli import u
from flext_core import r
from flext_infra._utilities._git.semantic_identity import (
    FlextInfraUtilitiesGitSemanticIdentityMixin,
)
from flext_infra._utilities._git.remote import canonical_origin_remote
from flext_infra.models import m

if TYPE_CHECKING:
    from flext_infra import p, t

_TAG_PREFIX = "attest/gates/v1"
_SSH_SIGNATURE_MARKER = "\n-----BEGIN SSH SIGNATURE-----"


class FlextInfraUtilitiesGitAttestationMixin(
    FlextInfraUtilitiesGitSemanticIdentityMixin
):
    """Create and verify immutable SSH-signed gate tags."""

    @staticmethod
    def _attestation_tag(commit_sha: str) -> str:
        return f"{_TAG_PREFIX}/{commit_sha}"

    @classmethod
    def _attestation_predicate(
        cls, request: m.Infra.GateAttestationCreateRequest
    ) -> p.Result[m.Infra.GateAttestationPredicate]:
        repo_root = Path(request.workspace).expanduser().resolve()
        identity = cls.git_identity(m.Infra.GitRepoRequest(repo_root=repo_root))
        if identity.failure:
            return r[m.Infra.GateAttestationPredicate].fail(
                identity.error or "failed to resolve Git identity"
            )
        repo = cls._repo(repo_root)
        if repo.is_dirty(untracked_files=False):
            return r[m.Infra.GateAttestationPredicate].fail(
                "gate attestation requires a clean committed worktree"
            )
        evidence_result = cls._run_gate_evidence(repo_root, request.gates)
        if evidence_result.failure:
            return r[m.Infra.GateAttestationPredicate].fail(
                evidence_result.error or "local gate execution failed"
            )
        toolchain = cls._toolchain_digest(repo_root)
        predicate = m.Infra.GateAttestationPredicate(
            schema_version="https://flext.sh/attestations/gates/v1",
            repository=canonical_origin_remote(identity.value.origin_remote or ""),
            commit_sha=identity.value.head_oid,
            tree_sha=repo.head.commit.tree.hexsha,
            signer=request.signer,
            toolchain_digest=toolchain,
            covered_gates=request.gates,
            commands=evidence_result.value,
        )
        return r[m.Infra.GateAttestationPredicate].ok(predicate)

    @classmethod
    def _run_gate_evidence(
        cls, repo_root: Path, gates: t.StrSequence
    ) -> p.Result[tuple[m.Infra.GateCommandEvidence, ...]]:
        evidence: list[m.Infra.GateCommandEvidence] = []
        for gate in gates:
            command = f"make {gate}" if gate != "gen" else "make gen WHAT=check"
            started = datetime.now(UTC)
            outcome = u.Cli.run_raw(command.split(), cwd=repo_root)
            completed = datetime.now(UTC)
            if outcome.failure:
                return r[tuple[m.Infra.GateCommandEvidence, ...]].fail(
                    outcome.error or f"local gate failed: {command}"
                )
            output = outcome.value
            digest_input = (
                f"command={command}\nexit_code=0\n"
                f"stdout={output.stdout}\nstderr={output.stderr}"
            )
            digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
            evidence.append(
                m.Infra.GateCommandEvidence(
                    gate=gate,
                    command=command,
                    cwd=str(repo_root),
                    exit_code=0,
                    result_digest=f"sha256:{digest}",
                    started_at=started.isoformat().replace("+00:00", "Z"),
                    completed_at=completed.isoformat().replace("+00:00", "Z"),
                )
            )
        return r[tuple[m.Infra.GateCommandEvidence, ...]].ok(tuple(evidence))

    @classmethod
    def _toolchain_digest(cls, repo_root: Path, commit_sha: str | None = None) -> str:
        repo = cls._repo(repo_root)
        commit = repo.commit(commit_sha) if commit_sha is not None else repo.head.commit
        names = (".mise.toml", ".python-version", "pyproject.toml", "uv.lock")
        tracked: list[str] = []
        for name in names:
            try:
                blob = commit.tree / name
            except KeyError:
                continue
            tracked.append(f"{name}:{blob.hexsha}")
        content = "\n".join(tracked)
        return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"

    @classmethod
    def git_create_gate_attestation(
        cls, request: m.Infra.GateAttestationCreateRequest
    ) -> p.Result[m.Infra.GateAttestationReport]:
        """Create the exact signed tag for a validated HEAD predicate."""
        repo_root = Path(request.workspace).expanduser().resolve()
        predicate_result = cls._attestation_predicate(request)
        if predicate_result.failure:
            return r[m.Infra.GateAttestationReport].fail(
                predicate_result.error or "invalid attestation predicate"
            )
        predicate = predicate_result.value
        tag = cls._attestation_tag(predicate.commit_sha)
        serialized = u.Cli.json_dumps(predicate.model_dump(mode="json"), sort_keys=True)
        if serialized.failure:
            return r[m.Infra.GateAttestationReport].fail(
                serialized.error or "failed to serialize attestation predicate"
            )
        repo = cls._repo(repo_root)
        try:
            if tag in {item.name for item in repo.tags}:
                return r[m.Infra.GateAttestationReport].fail(
                    f"attestation tag already exists: {tag}"
                )
            repo.git.tag("--sign", "--annotate", "--message", serialized.value, tag)
        except (GitCommandError, OSError, ValueError) as exc:
            return r[m.Infra.GateAttestationReport].fail(str(exc))
        return r[m.Infra.GateAttestationReport].ok(
            cls._attestation_report(tag, predicate)
        )

    @classmethod
    def git_verify_gate_attestation(
        cls, request: m.Infra.GateAttestationVerifyRequest
    ) -> p.Result[m.Infra.GateAttestationReport]:
        """Verify signature, signer, HEAD identity, and exact gate coverage."""
        repo_root = Path(request.workspace).expanduser().resolve()
        allowed = Path(request.allowed_signers).expanduser().resolve()
        if not allowed.is_file():
            return r[m.Infra.GateAttestationReport].fail(
                f"allowed_signers file not found: {allowed}"
            )
        commit_sha = request.commit_sha
        tag = cls._attestation_tag(commit_sha)
        verification = u.Cli.run_raw(
            [
                "git",
                "-c",
                "gpg.format=ssh",
                "-c",
                f"gpg.ssh.allowedSignersFile={allowed}",
                "verify-tag",
                "--raw",
                tag,
            ],
            cwd=repo_root,
        )
        if verification.failure:
            return r[m.Infra.GateAttestationReport].fail(
                verification.error or f"signature verification failed: {tag}"
            )
        predicate_result = cls._predicate_from_tag(repo_root, tag, commit_sha)
        if predicate_result.failure:
            return r[m.Infra.GateAttestationReport].fail(
                predicate_result.error or "invalid attestation tag"
            )
        predicate = predicate_result.value
        checked = cls._attestation_predicate_from_value(
            repo_root, predicate, commit_sha
        )
        if checked.failure:
            return r[m.Infra.GateAttestationReport].fail(
                checked.error or "attestation identity mismatch"
            )
        output = verification.value
        match = re.search(
            r'^Good "git" signature for (\S+) with ',
            f"{output.stdout}\n{output.stderr}",
            flags=re.MULTILINE,
        )
        if match is None or match.group(1) != predicate.signer:
            return r[m.Infra.GateAttestationReport].fail(
                f"verified signature does not identify signer: {predicate.signer}"
            )
        if tuple(request.expected_gates) != tuple(predicate.covered_gates):
            return r[m.Infra.GateAttestationReport].fail(
                "attestation gate coverage does not exactly match required gates"
            )
        if request.output is not None:
            serialized = u.Cli.json_dumps(
                predicate.model_dump(mode="json"), sort_keys=True
            )
            if serialized.failure:
                return r[m.Infra.GateAttestationReport].fail(
                    serialized.error or "failed to serialize verified predicate"
                )
            Path(request.output).expanduser().resolve().write_text(
                serialized.value, encoding="utf-8"
            )
        return r[m.Infra.GateAttestationReport].ok(
            cls._attestation_report(tag, predicate)
        )

    @classmethod
    def _predicate_from_tag(
        cls, repo_root: Path, tag: str, commit_sha: str
    ) -> p.Result[m.Infra.GateAttestationPredicate]:
        try:
            tag_ref = next(
                item for item in cls._repo(repo_root).tags if item.name == tag
            )
        except (OSError, StopIteration) as exc:
            return r[m.Infra.GateAttestationPredicate].fail(str(exc))
        if tag_ref.tag is None:
            return r[m.Infra.GateAttestationPredicate].fail(
                f"attestation is not an annotated tag: {tag}"
            )
        target = tag_ref.commit.hexsha
        if target != commit_sha:
            return r[m.Infra.GateAttestationPredicate].fail(
                "attestation tag target does not equal selected commit: "
                f"{target} != {commit_sha}"
            )
        message = tag_ref.tag.message.split(_SSH_SIGNATURE_MARKER, maxsplit=1)[0]
        parsed = u.Cli.json_loads(message)
        if parsed.failure:
            return r[m.Infra.GateAttestationPredicate].fail(
                parsed.error or "invalid attestation tag JSON"
            )
        try:
            predicate = m.Infra.GateAttestationPredicate.model_validate(parsed.value)
        except ValueError as exc:
            return r[m.Infra.GateAttestationPredicate].fail(str(exc))
        return r[m.Infra.GateAttestationPredicate].ok(predicate)

    @classmethod
    def _attestation_predicate_from_value(
        cls,
        repo_root: Path,
        predicate: m.Infra.GateAttestationPredicate,
        commit_sha: str,
    ) -> p.Result[bool]:
        identity = cls.git_identity(m.Infra.GitRepoRequest(repo_root=repo_root))
        if identity.failure:
            return r[bool].fail(identity.error or "failed to resolve Git identity")
        repo = cls._repo(repo_root)
        commit = repo.commit(commit_sha)
        tree_sha = commit.tree.hexsha
        actual_repository = canonical_origin_remote(identity.value.origin_remote or "")
        actual_toolchain = cls._toolchain_digest(repo_root, commit_sha)
        mismatches = tuple(
            field
            for field, matches in (
                ("repository", predicate.repository == actual_repository),
                ("commit", predicate.commit_sha == commit_sha),
                ("tree", predicate.tree_sha == tree_sha),
                ("toolchain", predicate.toolchain_digest == actual_toolchain),
            )
            if not matches
        )
        if mismatches:
            return r[bool].fail(
                "attestation does not match HEAD: " + ", ".join(mismatches)
            )
        return r[bool].ok(True)

    @staticmethod
    def _attestation_report(
        tag: str, predicate: m.Infra.GateAttestationPredicate
    ) -> m.Infra.GateAttestationReport:
        return m.Infra.GateAttestationReport(
            tag=tag,
            commit_sha=predicate.commit_sha,
            tree_sha=predicate.tree_sha,
            signer=predicate.signer,
            covered_gates=predicate.covered_gates,
        )


__all__: list[str] = ["FlextInfraUtilitiesGitAttestationMixin"]
