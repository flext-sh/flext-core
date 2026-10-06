# Copyright 2026 FLEXT
"""Hold broken tool releases inside a ``make upg`` Mise lock stage.

The ``upg`` lock stage already carries the bumped lock; a broken upstream
release fails its staged install. This probes the stage, parses the failing
tools, holds each at its newest installable release inside the staged
manifest, and re-proves the whole stage. The caller then retries the staged
install and publishes. The committed manifest never changes, so the next
``upg`` resolves the newest release afresh. Only ``make upg`` runs it.

It runs with a host Python before the project's virtual environment exists,
so it intentionally uses only stdlib.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


class MiseLockConverge:
    """Hold every failing tool of a staged lock at its newest installable release."""

    FIXED_ENVIRONMENT = (
        ("GIT_CONFIG_NOSYSTEM", "1"),
        ("GIT_TERMINAL_PROMPT", "0"),
        ("LANG", "C"),
        ("LC_ALL", "C"),
        ("MISE_SAFE", "1"),
        ("MISE_PARANOID", "true"),
        ("MISE_NO_ENV", "1"),
        ("MISE_NO_HOOKS", "1"),
        ("MISE_AUTO_ENV", "false"),
        ("MISE_AUTO_INSTALL", "false"),
        ("MISE_EXEC_AUTO_INSTALL", "false"),
        ("MISE_TASK_RUN_AUTO_INSTALL", "false"),
        ("MISE_AUTO_UPDATE", "false"),
        ("MISE_HTTP_RETRIES", "0"),
        ("MISE_NETRC", "false"),
        ("MISE_NOT_FOUND_AUTO_INSTALL", "false"),
        ("MISE_NOT_FOUND_SYSTEM_FALLBACK", "false"),
        ("MISE_OVERRIDE_CONFIG_FILENAMES", ".mise.toml"),
        ("MISE_OVERRIDE_TOOL_VERSIONS_FILENAMES", "none"),
        ("MISE_GITHUB_GH_CLI_TOKENS", "false"),
        ("MISE_GITHUB_USE_GIT_CREDENTIALS", "false"),
        ("MISE_GITHUB_OAUTH_CLIENT_ID", ""),
        ("MISE_GITHUB_OAUTH_EXPORT_ENV", ""),
        ("MISE_GITHUB_OAUTH_OPEN_BROWSER", "false"),
        ("MISE_LOCKFILE", "true"),
        ("MISE_LOCKED", "true"),
        (
            "MISE_LOCKFILE_PLATFORMS",
            "linux-x64,linux-x64-musl,linux-arm64,macos-x64,macos-arm64,windows-x64",
        ),
        ("MISE_MINIMUM_RELEASE_AGE", "7d"),
        ("MISE_NPM_PACKAGE_MANAGER", "bun"),
    )
    TRANSIENT_ENVIRONMENT = (
        ("HOME", "home"),
        ("USERPROFILE", "home"),
        ("APPDATA", "appdata"),
        ("LOCALAPPDATA", "appdata"),
        ("XDG_CONFIG_HOME", "xdg-config"),
        ("XDG_DATA_HOME", "xdg-data"),
        ("XDG_CACHE_HOME", "xdg-cache"),
        ("XDG_STATE_HOME", "xdg-state"),
        ("NETRC", "netrc"),
        ("GIT_CONFIG_GLOBAL", "gitconfig"),
        ("MISE_NETRC_FILE", "netrc"),
        ("MISE_GLOBAL_CONFIG_FILE", "global-config.toml"),
        ("MISE_CONFIG_DIR", "config"),
        ("MISE_TMP_DIR", "tmp"),
        ("MISE_GLOBAL_CONFIG_ROOT", "."),
        ("MISE_SYSTEM_CONFIG_DIR", "system-config"),
        ("MISE_SYSTEM_CONFIG_FILE", "system-config/config.toml"),
        ("MISE_SYSTEM_DATA_DIR", "system-data"),
        ("MISE_SYSTEM_INSTALLS_DIR", "system-installs"),
        ("MISE_SYSTEM_SHIMS_DIR", "system-shims"),
        ("TMPDIR", "tmp"),
        ("TMP", "tmp"),
        ("TEMP", "tmp"),
    )
    PERSISTENT_ENVIRONMENT = (
        ("MISE_DATA_DIR", "."),
        ("MISE_CACHE_DIR", "cache"),
        ("MISE_STATE_DIR", "state"),
        ("MISE_INSTALLS_DIR", "installs"),
        ("MISE_SHIMS_DIR", "shims"),
        ("UV_CACHE_DIR", "uv-cache"),
    )
    EMPTY_FILES = (
        "global-config.toml",
        "system-config/config.toml",
        "gitconfig",
        "netrc",
    )
    PASSTHROUGH_ENVIRONMENT = (
        "PATH",
        "COMSPEC",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "MISE_GITHUB_TOKEN",
        "MISE_HTTP_TIMEOUT",
        "FLEXT_MYPY_PROFILE_OUTPUT",
        "MISE_VERSION",
    )
    RUNTIME_INSTALL_RELATIVE_TEMPLATE = "bootstrap/mise-{release}"
    CANDIDATE_LIMIT = 8

    @classmethod
    def _runtime(cls, storage: Path, release: str) -> Path:
        """Locate the pinned Mise runtime the bootstrap installed."""
        relative = cls.RUNTIME_INSTALL_RELATIVE_TEMPLATE.format(
            release=release.removeprefix("v"),
        )
        runtime = storage / relative
        if os.name == "nt":
            runtime = runtime.with_name(f"{runtime.name}.exe")
        if not (runtime.is_file() and os.access(runtime, os.X_OK)):
            msg = f"missing pinned Mise runtime {runtime}; run make upg"
            raise ValueError(msg)
        return runtime

    @classmethod
    def _environment(cls, storage: Path, stage: Path, scratch: Path) -> dict[str, str]:
        """Build the isolated environment the bootstrap recipe runs Mise in."""
        for _name, relative in cls.TRANSIENT_ENVIRONMENT:
            if relative not in cls.EMPTY_FILES:
                (scratch / relative).mkdir(parents=True, exist_ok=True)
        for relative in cls.EMPTY_FILES:
            (scratch / relative).parent.mkdir(parents=True, exist_ok=True)
            (scratch / relative).write_bytes(b"")
        environment = dict(cls.FIXED_ENVIRONMENT)
        environment.update(
            (name, str(scratch / relative))
            for name, relative in cls.TRANSIENT_ENVIRONMENT
        )
        environment.update(
            (name, str(storage if relative == "." else storage / relative))
            for name, relative in cls.PERSISTENT_ENVIRONMENT
        )
        environment.update(
            (name, os.environ[name])
            for name in cls.PASSTHROUGH_ENVIRONMENT
            if os.environ.get(name)
        )
        environment["GIT_CEILING_DIRECTORIES"] = str(stage.parent)
        environment["MISE_CEILING_PATHS"] = str(stage.parent)
        environment["MISE_TRUSTED_CONFIG_PATHS"] = str(stage)
        return environment

    @staticmethod
    def _run(runtime: Path, arguments: list[str], environment: dict[str, str]) -> str:
        """Run one isolated Mise command; warnings and failures escape loudly."""
        completed = subprocess.run(
            [str(runtime), *arguments],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        diagnostics = completed.stdout + completed.stderr
        if completed.returncode != 0:
            sys.stderr.write(diagnostics)
            msg = f"Mise exited {completed.returncode}: {' '.join(arguments)}\n{diagnostics.strip()}"
            raise ValueError(
                msg,
            )
        if "mise WARN" in diagnostics:
            sys.stderr.write(diagnostics)
            msg = f"Mise warned during {' '.join(arguments)}; converge stopped"
            raise ValueError(msg)
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        return completed.stdout.strip()

    @staticmethod
    def _probe(
        runtime: Path, stage: Path, environment: dict[str, str]
    ) -> tuple[bool, str]:
        """Prove the staged lock installs without mutating tools."""
        completed = subprocess.run(
            [str(runtime), "-C", str(stage), "install", "--dry-run"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode == 0, completed.stdout + completed.stderr

    @staticmethod
    def failing_install_tools(probe_output: str) -> list[tuple[str, str]]:
        """Extract the ``selector@version`` pairs a failed install probe named."""
        tools: list[tuple[str, str]] = []
        marker = "Failed to install tools:"
        for line in probe_output.splitlines():
            if marker not in line:
                continue
            for item in line.split(marker, 1)[1].split(","):
                selector, _, version = item.strip().rpartition("@")
                version = version.strip().lstrip("v")
                if selector and version and version[0].isdigit():
                    tools.append((selector, version))
        if not tools:
            msg = f"staged install failed but named no failing tool: {probe_output.strip()[:400]}"
            raise ValueError(
                msg,
            )
        return tools

    @classmethod
    def release_candidates(cls, listing: str, failed_version: str) -> list[str]:
        """List releases of an ``ls-remote`` listing strictly older than the failed one."""

        def release_key(version: str) -> tuple[int, ...] | None:
            try:
                return tuple(int(part) for part in version.split("."))
            except ValueError:
                return None

        failed = release_key(failed_version)
        candidates: list[str] = []
        for line in listing.splitlines():
            version = line.strip().lstrip("v")
            parsed = release_key(version)
            if parsed is None or (failed is not None and parsed >= failed):
                continue
            candidates.append(version)
        return candidates[: cls.CANDIDATE_LIMIT]

    @staticmethod
    def hold_manifest_version(manifest: Path, selector: str, version: str) -> None:
        """Rewrite one tool's declared version inside the staged manifest copy."""
        lines = manifest.read_text(encoding="utf-8").splitlines(keepends=True)
        headers = (f'[tools."{selector}"]', f"[tools.{selector}]")
        in_section = False
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[tools."):
                in_section = stripped in headers
                continue
            if in_section and stripped.startswith("version") and "=" in stripped:
                lines[index] = f'version = "{version}"\n'
                manifest.write_text("".join(lines), encoding="utf-8")
                return
        msg = f"Mise manifest has no declared version to hold: {selector}"
        raise ValueError(msg)

    @staticmethod
    def staged_manifest(stage: Path) -> Path:
        """Resolve the staged manifest, refusing a path that escapes the stage.

        The stage directory arrives from the command line, so the manifest
        write is guarded: a symlinked or otherwise relocated ``.mise.toml``
        that resolves outside the declared stage stops converge loud instead
        of rewriting an unrelated file.
        """
        manifest = (stage / ".mise.toml").resolve()
        if not manifest.is_relative_to(stage.resolve()):
            msg = f"staged manifest escapes the stage: {manifest}"
            raise ValueError(msg)
        return manifest

    @classmethod
    def _hold(
        cls,
        runtime: Path,
        stage: Path,
        environment: dict[str, str],
        selector: str,
        failed_version: str,
    ) -> str:
        """Hold one failing tool at its newest release that installs in the stage."""
        listing = cls._run(runtime, ["ls-remote", selector], environment)
        manifest = cls.staged_manifest(stage)
        for candidate in cls.release_candidates(listing, failed_version):
            cls.hold_manifest_version(manifest, selector, candidate)
            try:
                cls._run(runtime, ["-C", str(stage), "lock"], environment)
            except ValueError as error:
                if "refusing to replace locked version" in str(error):
                    continue
                raise
            if cls._probe(runtime, stage, environment)[0]:
                return candidate
        msg = (
            f"no installable release found below {failed_version} for {selector};"
            " upgrade needs an operator decision"
        )
        raise ValueError(
            msg,
        )

    @classmethod
    def converge(cls, storage: Path, stage: Path, release: str) -> None:
        """Hold failing tools of the staged lock, then re-prove the whole stage."""
        if not (stage / ".mise.toml").is_file():
            msg = f"missing staged Mise manifest: {stage / '.mise.toml'}"
            raise ValueError(msg)
        runtime = cls._runtime(storage, release)
        scratch = Path(tempfile.mkdtemp(prefix="mise-converge."))
        try:
            environment = cls._environment(storage, stage, scratch)
            satisfied, probe_output = cls._probe(runtime, stage, environment)
            if satisfied:
                print("converge: staged lock installs; nothing to hold")
                return
            holds: dict[str, str] = {}
            for selector, failed_version in cls.failing_install_tools(probe_output):
                holds[selector] = cls._hold(
                    runtime, stage, environment, selector, failed_version
                )
                print(
                    f"hold: {selector} held at {holds[selector]}: release {failed_version}"
                    " failed install; the next upg retries the newest release",
                )
            if not cls._probe(runtime, stage, environment)[0]:
                msg = f"converge: held lock still fails install: {sorted(holds)}"
                raise ValueError(msg)
            print(f"converge: staged lock installs with holds {sorted(holds)}")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    @classmethod
    def main(cls, arguments: list[str]) -> int:
        if len(arguments) != 3:
            msg = "usage: mise-lock-converge.py STORAGE STAGE RELEASE"
            raise ValueError(msg)
        cls.converge(
            Path(arguments[0]).absolute(), Path(arguments[1]).absolute(), arguments[2]
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(MiseLockConverge.main(sys.argv[1:]))
