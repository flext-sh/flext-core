"""Fail-closed validation of workspace editable installation provenance."""

from __future__ import annotations

import re
from importlib.metadata import distributions
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from flext_core import r
from flext_infra import m, u
from flext_infra.workspace.detector import FlextInfraWorkspaceDetector

if TYPE_CHECKING:
    from flext_infra import p, t


class FlextInfraWorkspaceEnvironmentProvenance:
    """Validate that every declared editable resolves to the live workspace."""

    @classmethod
    def execute_request(
        cls, request: p.Infra.WorkspaceEnvironmentRequest
    ) -> p.Result[int]:
        """Validate one CLI request without mutating the environment."""
        return cls.validate(request.repository_root)

    @classmethod
    def validate(
        cls, repository_root: Path, *, metadata_paths: t.StrSequence | None = None
    ) -> p.Result[int]:
        """Validate PEP 610 and editable path metadata for active members."""
        resolved_root = repository_root.resolve()
        workspace_result = FlextInfraWorkspaceDetector.load_workspace_spec(
            resolved_root
        )
        if workspace_result.failure:
            return r[int].fail(
                workspace_result.error or "workspace topology validation failed"
            )
        repositories = tuple(
            repository
            for repository in workspace_result.value.declared_repositories
            if repository.package and repository.editable
        )
        validated = 0
        for repository in repositories:
            matches = (
                tuple(distributions(name=repository.distribution))
                if metadata_paths is None
                else tuple(
                    distributions(
                        name=repository.distribution, path=list(metadata_paths)
                    )
                )
            )
            if len(matches) != 1:
                return r[int].fail(
                    "editable provenance distribution count mismatch: "
                    f"distribution={repository.distribution} expected=1 "
                    f"actual={len(matches)}"
                )
            distribution = matches[0]
            expected_root = (resolved_root / repository.path).resolve()
            direct_url_result = cls._validate_direct_url(
                repository.distribution,
                distribution.read_text("direct_url.json"),
                expected_root,
            )
            if direct_url_result.failure:
                return direct_url_result
            files = distribution.files
            if files is None:
                return r[int].fail(
                    "editable provenance has no installed file inventory: "
                    f"distribution={repository.distribution}"
                )
            pth_files = tuple(
                Path(str(distribution.locate_file(file)))
                for file in files
                if str(file).endswith(".pth")
            )
            if len(pth_files) != 1:
                return r[int].fail(
                    "editable provenance pth count mismatch: "
                    f"distribution={repository.distribution} expected=1 "
                    f"actual={len(pth_files)}"
                )
            pth_result = cls._validate_pth(
                repository.distribution, pth_files[0], expected_root
            )
            if pth_result.failure:
                return pth_result
            validated += 1
        return r[int].ok(validated)

    @classmethod
    def _validate_direct_url(
        cls, distribution: str, raw_payload: str | None, expected_root: Path
    ) -> p.Result[int]:
        """Validate one PEP 610 payload against the declared member root."""
        if raw_payload is None:
            return r[int].fail(
                "editable provenance missing direct_url.json: "
                f"distribution={distribution} expected={expected_root}"
            )
        try:
            payload = m.Infra.EditableDirectUrl.model_validate_json(
                raw_payload, strict=True
            )
        except ValueError as exc:
            return r[int].fail_op(
                f"editable provenance direct_url validation ({distribution})", exc
            )
        parsed = urlparse(payload.url)
        if parsed.scheme != "file" or not payload.dir_info.editable:
            return r[int].fail(
                "editable provenance is not an editable file URL: "
                f"distribution={distribution} url={payload.url}"
            )
        actual_root = Path(url2pathname(unquote(parsed.path))).resolve()
        if actual_root != expected_root:
            return r[int].fail(
                "editable provenance direct_url mismatch: "
                f"distribution={distribution} expected={expected_root} "
                f"actual={actual_root}"
            )
        return r[int].ok(1)

    @classmethod
    def _validate_pth(
        cls, distribution: str, pth_file: Path, expected_root: Path
    ) -> p.Result[int]:
        """Validate the distribution-owned editable path file."""
        read_result = u.Cli.files_read_text(pth_file)
        if read_result.failure:
            return r[int].fail(
                read_result.error or f"editable provenance pth read failed: {pth_file}"
            )
        entries = tuple(
            line.strip()
            for line in read_result.value.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if not entries:
            return r[int].fail(
                "editable provenance pth is empty: "
                f"distribution={distribution} path={pth_file}"
            )
        for entry in entries:
            if re.match(r"^import\s", entry) or not Path(entry).is_absolute():
                return r[int].fail(
                    "editable provenance pth entry is not an absolute source path: "
                    f"distribution={distribution} entry={entry}"
                )
            actual_source = Path(entry).resolve()
            if not actual_source.is_relative_to(expected_root):
                return r[int].fail(
                    "editable provenance pth mismatch: "
                    f"distribution={distribution} expected_root={expected_root} "
                    f"actual={actual_source}"
                )
        return r[int].ok(1)


__all__: list[str] = ["FlextInfraWorkspaceEnvironmentProvenance"]
