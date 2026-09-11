"""Docker container control facade for FLEXT test infrastructure."""

from __future__ import annotations

import os
import socket
import time
from collections.abc import Generator, MutableSet
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Self, override

import pytest
from docker import DockerClient as DockerSDKClient, from_env as docker_from_env
from docker.constants import DEFAULT_DOCKER_API_VERSION
from docker.errors import DockerException, NotFound
from docker.transport import UnixHTTPAdapter
from python_on_whales import DockerClient as WhalesDockerClient
from python_on_whales.exceptions import DockerException as WhalesDockerException

from flext_tests import c, m, p, r, s, t, u

if TYPE_CHECKING:
    from docker.models.containers import Container


class FlextTestsDocker(s[m.Tests.ContainerInfo]):
    """Manage Docker containers for FLEXT tests."""

    docker: ClassVar[WhalesDockerClient] = WhalesDockerClient(client_type="docker")

    workspace_root: Annotated[
        Path, u.Field(description="Workspace root used to resolve compose files.")
    ] = u.Field(default_factory=Path.cwd)

    worker_id: Annotated[
        str, u.Field(description="Worker identifier used to isolate persisted state.")
    ] = "master"

    docker_client: Annotated[
        DockerSDKClient | None,
        u.Field(exclude=True, description="Cached Docker SDK client instance."),
    ] = None

    client_error: Annotated[
        str | None,
        u.Field(exclude=True, description="Last Docker client initialization error."),
    ] = None

    dirty_container_names: Annotated[
        MutableSet[str],
        u.Field(exclude=True, description="Tracked dirty containers for the worker."),
    ] = u.Field(default_factory=set)

    state_file_path: Annotated[
        Path | None,
        u.Field(exclude=True, description="Persistent dirty container state file."),
    ] = None

    target_config: Annotated[
        m.Tests.ContainerConfig | None,
        u.Field(description="Configured Docker target used by the public DSL."),
    ] = None

    @staticmethod
    def ci_disables_docker() -> bool:
        """True when Make CI token is exact CI=Y (not GitHub CI=true)."""
        return os.environ.get(c.Tests.ENV_CI) == c.Tests.CI_MAKE_VALUE

    @classmethod
    def skip_if_ci_disables_docker(cls) -> None:
        """Fail-closed skip for any Docker lifecycle entry under CI=Y."""
        if cls.ci_disables_docker():
            pytest.skip(c.Tests.DOCKER_CI_SKIP_REASON)

    @staticmethod
    def _resolve_shared_target_config(
        container_name: str, workspace_root: Path
    ) -> m.Tests.ContainerConfig:
        """Resolve one shared-container entry into the canonical target config."""
        settings = c.Tests.SHARED_CONTAINERS.get(container_name)
        if settings is None:
            msg = f"Unknown shared container: {container_name}"
            raise ValueError(msg)
        compose_file_raw = settings.get("compose_file")
        if not compose_file_raw:
            msg = f"Shared container '{container_name}' missing compose_file"
            raise ValueError(msg)
        target = m.Tests.ContainerConfig.model_validate({
            **settings,
            "compose_file": Path(str(compose_file_raw)),
        })
        compose_path = Path(str(compose_file_raw))
        if not compose_path.is_absolute():
            compose_path = workspace_root / compose_path
        return target.model_copy(
            update={"container_name": container_name, "compose_file": compose_path}
        )

    @staticmethod
    def _resolve_readiness_port(
        target: m.Tests.ContainerConfig, info: m.Tests.ContainerInfo
    ) -> int | None:
        """Resolve the host port used by a target readiness check."""
        if target.port is None:
            return None
        target_port_str = str(target.port)
        for container_port, host_port in info.ports.items():
            if container_port.startswith(f"{target_port_str}/"):
                return int(host_port)
            if host_port == target_port_str:
                return int(host_port)
        return int(target_port_str) if target_port_str.isdigit() else None

    @staticmethod
    def _extract_host_port(bindings: t.SequenceOf[t.StrMapping] | None) -> str:
        """Extract the first Docker HostPort value from normalized bindings."""
        if not bindings:
            return ""
        return bindings[0].get("HostPort", "")

    @override
    def model_post_init(self, __context: t.JsonValue | None, /) -> None:
        """Initialize private runtime state after model validation."""
        super().model_post_init(__context)
        self.state_file_path = (
            Path.home() / ".flext" / f"docker_state_{self.worker_id}.json"
        )
        self._load_dirty_state()
        _ = self.client

    @property
    def client(self) -> DockerSDKClient | None:
        """Docker client with lazy initialization."""
        if self.docker_client is None and self.client_error is None:
            client: DockerSDKClient | None = None
            try:
                client = docker_from_env(version=DEFAULT_DOCKER_API_VERSION)
                adapter = client.api.get_adapter(client.api.base_url)
                if (
                    isinstance(adapter, UnixHTTPAdapter)
                    and not Path(adapter.socket_path).exists()
                ):
                    raise FileNotFoundError(adapter.socket_path)
                _ = client.ping()
            except (DockerException, OSError, TypeError, ValueError) as error:
                if client is not None:
                    client.close()
                self.logger.exception(
                    "Failed to initialize Docker client", error=str(error)
                )
                self.client_error = str(error)
            else:
                self.docker_client = client
                self.client_error = None
        return self.docker_client

    @property
    def dirty_containers(self) -> t.StrSequence:
        """Dirty container names."""
        return tuple(self.dirty_container_names)

    def container_dirty(self, container_name: str) -> bool:
        """Whether a container is marked as dirty."""
        return container_name in self.dirty_container_names

    def mark_container_clean(self, container_name: str) -> p.Result[bool]:
        """Mark a container as clean after successful recreation."""
        try:
            self.dirty_container_names.discard(container_name)
            self._save_dirty_state()
            self.logger.info("Container marked clean", container=container_name)
            return r[bool].ok(value=True)
        except c.EXC_OS_TYPE as exc:
            return r[bool].fail(f"Failed to mark clean: {exc}")

    def mark_container_dirty(self, container_name: str) -> p.Result[bool]:
        """Mark a container as dirty for recreation on next use."""
        try:
            self.dirty_container_names.add(container_name)
            self._save_dirty_state()
            self.logger.info("Container marked dirty", container=container_name)
            return r[bool].ok(value=True)
        except c.EXC_OS_TYPE as exc:
            return r[bool].fail(f"Failed to mark dirty: {exc}")

    def _load_dirty_state(self) -> None:
        """Load dirty container state from persistent storage."""
        state_file = self.state_file_path
        if state_file is None or not state_file.exists():
            return
        read = u.Cli.files_read_text(state_file)
        if read.failure:
            self.logger.warning("Failed to load dirty state", error=read.error)
            self.dirty_container_names = set[str]()
            return
        try:
            state_raw: t.MappingKV[str, t.StrSequence] = (
                t.Tests.STR_SEQUENCE_MAPPING_ADAPTER.validate_json(read.value)
            )
            self.dirty_container_names = set(state_raw.get("dirty_containers", ()))
        except c.EXC_KEY_OS_TYPE_VALUE as exc:
            self.logger.warning("Failed to load dirty state", error=str(exc))
            self.dirty_container_names = set[str]()

    def _save_dirty_state(self) -> None:
        """Save dirty container state to persistent storage."""
        state_file = self.state_file_path
        if state_file is None:
            return
        data: t.MappingKV[str, t.StrSequence] = {
            "dirty_containers": tuple(self.dirty_container_names)
        }
        write = u.Cli.json_write(state_file, data)
        if write.failure:
            self.logger.warning("Failed to save dirty state", error=write.error)

    def compose_down(self, compose_file: str) -> p.Result[str]:
        """Stop services using docker-compose via python_on_whales."""
        self.skip_if_ci_disables_docker()
        compose_path = self._compose_path(compose_file)
        try:
            self._run_compose_down(compose_path)
        except self._compose_exception_types() as exc:
            return r[str].fail_op("Compose down", exc)
        return r[str].ok("Compose down successful")

    def compose_up(
        self,
        compose_file: str,
        service: str | None = None,
        *,
        force_recreate: bool = False,
    ) -> p.Result[str]:
        """Start services using docker-compose via python_on_whales."""
        self.skip_if_ci_disables_docker()
        compose_path = self._compose_path(compose_file)
        try:
            cleanup_result = self._run_compose_up(
                compose_path, service, force_recreate=force_recreate
            )
        except self._compose_exception_types() as exc:
            self.logger.exception("Compose up failed")
            return r[str].fail_op("Compose up", exc)
        if cleanup_result.failure:
            return cleanup_result
        return r[str].ok("Compose up successful")

    def _compose_path(self, compose_file: str) -> Path:
        """Resolve a compose path against the configured workspace root."""
        compose_path = Path(compose_file)
        return (
            compose_path
            if compose_path.is_absolute()
            else self.workspace_root / compose_file
        )

    @staticmethod
    def _compose_exception_types() -> tuple[type[Exception], ...]:
        """Return compose exception types handled by this adapter."""
        return (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            WhalesDockerException,
        )

    @staticmethod
    def compose_project_name(compose_path: Path) -> str:
        """Derive one compose project per compose file.

        Compose derives the project name from the directory when none is given,
        so every file under the shared ``docker/`` directory lands in the same
        project. ``remove_orphans`` then treats a sibling suite's container as
        garbage and deletes it, which makes parallel test runs kill each other's
        databases. Binding each compose file to its own project keeps
        ``remove_orphans`` scoped to the services that file actually declares.
        """
        return compose_path.stem.replace(".", "-").replace("_", "-")

    @contextmanager
    def _compose_binding(self, compose_path: Path) -> Generator[None]:
        """Bind compose file and its dedicated project for one operation."""
        config = self.docker.client_config
        original_files = config.compose_files
        original_project = config.compose_project_name
        try:
            config.compose_files = [str(compose_path)]
            config.compose_project_name = self.compose_project_name(compose_path)
            yield
        finally:
            config.compose_files = original_files
            config.compose_project_name = original_project

    def _run_compose_down(self, compose_path: Path) -> None:
        """Run compose down with temporary compose-file binding."""
        with self._compose_binding(compose_path):
            self.docker.compose.down(volumes=True, remove_orphans=True)

    def _run_compose_up(
        self, compose_path: Path, service: str | None, *, force_recreate: bool
    ) -> p.Result[str]:
        """Run compose up with temporary compose-file binding."""
        with self._compose_binding(compose_path):
            if force_recreate:
                down_result = self._compose_down_current_file()
                if down_result.failure:
                    return down_result
            # python-on-whales treats an EMPTY service list as a no-op and
            # returns without running compose at all; only None means "every
            # service in the file". Passing [] here silently started nothing.
            services = [service] if service else None
            # `wait` blocks until every started service reports healthy (or
            # running, when it declares no healthcheck). Without it compose
            # returns as soon as the containers are created, so a cold boot
            # hands back a stack whose database is still initializing and every
            # readiness probe races first-run setup.
            self.docker.compose.up(
                services=services, detach=True, remove_orphans=True, wait=True
            )
        return r[str].ok("Compose up successful")

    def _compose_down_current_file(self) -> p.Result[str]:
        """Run compose down for the currently configured compose file."""
        try:
            self.docker.compose.down(remove_orphans=True, volumes=True)
        except self._compose_exception_types() as exc:
            self.logger.warning("Compose recreate cleanup failed", error=str(exc))
            return r[str].fail_op("Compose recreate cleanup", exc)
        return r[str].ok("Compose recreate cleanup successful")

    def fetch_container_info(
        self, container_name: str
    ) -> p.Result[m.Tests.ContainerInfo]:
        """Fetch container information."""
        client = self.client
        if client is None:
            error = self.client_error or "Docker daemon unavailable"
            return r[m.Tests.ContainerInfo].fail(error)
        try:
            container = client.containers.get(container_name)
        except NotFound:
            return r[m.Tests.ContainerInfo].fail(
                f"Container {container_name} not found"
            )
        except c.EXC_BROAD_RUNTIME as exc:
            return r[m.Tests.ContainerInfo].fail(str(exc))
        return r[m.Tests.ContainerInfo].ok(
            self._container_info_from_sdk(container_name, container)
        )

    def fetch_container_status(
        self, container_name: str
    ) -> p.Result[m.Tests.ContainerInfo]:
        """Fetch container status."""
        return self.fetch_container_info(container_name)

    def start_existing_container(self, container_name: str) -> p.Result[bool]:
        """Start an existing stopped container by name."""
        client = self.client
        if client is None:
            error = self.client_error or "Docker daemon unavailable"
            return r[bool].fail(error)
        try:
            container = client.containers.get(container_name)
        except NotFound:
            return r[bool].fail(f"Container {container_name} not found")
        except (DockerException, OSError, RuntimeError, AttributeError) as exc:
            return r[bool].fail(f"Failed to inspect container {container_name}: {exc}")
        return self._start_sdk_container(container_name, container)

    def start_compose_stack(
        self, compose_file: str, network_name: str | None = None
    ) -> p.Result[str]:
        """Start a Docker Compose stack."""
        _ = network_name
        result = self.compose_up(compose_file)
        if result.failure:
            return r[str].fail_op("Stack start", result.error)
        return r[str].ok("Stack started successfully")

    def wait_for_port_ready(
        self, host: str, port: int, max_wait: int | None = None
    ) -> p.Result[bool]:
        """Wait until a TCP port is accepting connections.

        Uses ``max_wait`` when provided (shared-container startup budgets).
        When omitted, uses ``DOCKER_PROBE_MAX_WAIT_SECONDS`` so quick probes
        fail closed as a Result instead of hanging into pytest case-timeouts.
        """
        probe_budget = (
            max_wait if max_wait is not None else c.Tests.DOCKER_PROBE_MAX_WAIT_SECONDS
        )
        waited = 0.0
        while waited < probe_budget:
            try:
                with socket.create_connection((host, port), timeout=1):
                    return r[bool].ok(value=True)
            except (ConnectionRefusedError, TimeoutError, OSError):
                time.sleep(0.5)
                waited += 0.5
        return r[bool].fail(
            f"TCP {host}:{port} not ready within {probe_budget}s probe budget"
        )

    def _container_info_from_sdk(
        self, container_name: str, container: Container
    ) -> m.Tests.ContainerInfo:
        """Build canonical container info from a Docker SDK container."""
        ports_raw: t.MappingKV[str, t.Tests.TestobjectSerializable] = (
            t.Tests.TESTOBJECT_SERIALIZABLE_MAPPING_ADAPTER.validate_python(
                container.ports
            )
        )
        ports: t.MutableStrMapping = {}
        for container_port, host_bindings in ports_raw.items():
            normalized_bindings = t.Tests.STR_MAPPING_SEQUENCE_ADAPTER.validate_python(
                host_bindings
            )
            host_port = self._extract_host_port(normalized_bindings)
            if host_port:
                ports[container_port] = host_port
        image_obj = container.image
        image_tags_raw = image_obj.tags if image_obj is not None else ()
        image_tags = list(image_tags_raw)
        return m.Tests.ContainerInfo(
            name=container_name,
            status=c.Tests.ContainerStatus(container.status),
            ports=ports,
            image=image_tags[0] if image_tags else "",
            container_id=str(container.id),
        )

    @staticmethod
    def _start_sdk_container(
        container_name: str, container: Container
    ) -> p.Result[bool]:
        """Start a Docker SDK container when it is not already running."""
        try:
            if container.status == "running":
                return r[bool].ok(value=True)
            container.start()
        except (DockerException, OSError, RuntimeError, AttributeError) as exc:
            return r[bool].fail(f"Failed to start container {container_name}: {exc}")
        return r[bool].ok(value=True)

    @classmethod
    def shared(
        cls,
        container_name: str,
        *,
        workspace_root: Path | None = None,
        worker_id: str | None = None,
    ) -> Self:
        """Build a DSL-configured service from a shared container constant."""
        resolved_root = workspace_root or Path.cwd()
        return cls(
            workspace_root=resolved_root,
            worker_id=worker_id or "master",
            target_config=cls._resolve_shared_target_config(
                container_name, resolved_root
            ),
        )

    @classmethod
    def compose(
        cls,
        compose_file: str | Path,
        *,
        target: m.Tests.ContainerConfig | None = None,
        workspace_root: Path | None = None,
    ) -> Self:
        """Build a DSL-configured service for an explicit compose target."""
        resolved_root = workspace_root or Path.cwd()
        compose_path = Path(compose_file)
        if not compose_path.is_absolute():
            compose_path = resolved_root / compose_path
        base_target = target or m.Tests.ContainerConfig()
        return cls(
            workspace_root=resolved_root,
            worker_id="master",
            target_config=base_target.model_copy(update={"compose_file": compose_path}),
        )

    @classmethod
    def stack(
        cls,
        compose_file: str | Path,
        *,
        target: m.Tests.ContainerConfig | None = None,
        workspace_root: Path | None = None,
    ) -> Self:
        """Build a DSL-configured service for a compose stack target."""
        return cls.compose(compose_file, target=target, workspace_root=workspace_root)

    def up(self) -> p.Result[str]:
        """Start the configured compose target using the DSL state."""
        target = self.target_config
        if target is None:
            return r[str].fail(
                "Docker target not configured. Use tk.shared(...), tk.compose(...), or tk.stack(...)."
            )
        if target.compose_file is None:
            return r[str].fail("Docker target has no compose file configured.")
        return self.compose_up(
            str(target.compose_file),
            service=target.service or None,
            force_recreate=target.force_recreate,
        )

    def down(self) -> p.Result[str]:
        """Stop the configured compose target using the DSL state."""
        target = self.target_config
        if target is None:
            return r[str].fail(
                "Docker target not configured. Use tk.shared(...), tk.compose(...), or tk.stack(...)."
            )
        if target.compose_file is None:
            return r[str].fail("Docker target has no compose file configured.")
        return self.compose_down(str(target.compose_file))

    def ready(
        self, *, port: int | None = None, max_wait: int | None = None
    ) -> p.Result[bool]:
        """Run a readiness check against the configured target."""
        target = self.target_config
        if target is None:
            return r[bool].fail(
                "Docker target not configured. Use tk.shared(...), tk.compose(...), or tk.stack(...)."
            )
        resolved_port = target.port if port is None else port
        if resolved_port is None:
            return r[bool].fail(
                f"Docker target {target.container_name} has no configured readiness port."
            )
        return self.wait_for_port_ready(
            target.host,
            resolved_port,
            max_wait=target.startup_timeout if max_wait is None else max_wait,
        )

    def cleanup_dirty_containers(self) -> p.Result[t.StrSequence]:
        """Clean up all dirty containers by recreating them with fresh volumes."""
        cleaned: list[str] = []
        for container_name in list(self.dirty_container_names):
            if container_name not in c.Tests.SHARED_CONTAINERS:
                self.logger.warning(
                    "Removing stale dirty container entry", container=container_name
                )
                _ = self.mark_container_clean(container_name)
                continue
            target = self._resolve_shared_target_config(
                container_name, self.workspace_root
            )
            self.logger.info("Recreating dirty container", container=container_name)
            _ = self.compose_down(str(target.compose_file))
            result = self.compose_up(
                str(target.compose_file), target.service, force_recreate=True
            )
            if result.success:
                _ = self.mark_container_clean(container_name)
                cleaned.append(container_name)
        return r[t.StrSequence].ok(tuple(cleaned))

    @override
    def execute(self) -> p.Result[m.Tests.ContainerInfo]:
        """Ensure the configured container is available with a single DSL call."""
        self.skip_if_ci_disables_docker()
        target = self.target_config
        if target is None:
            return r[m.Tests.ContainerInfo].fail(
                "Docker target not configured. Use tk.shared(...).execute() or tk.compose(...).execute()."
            )
        if not target.container_name:
            return r[m.Tests.ContainerInfo].fail(
                "Docker target has no inspection container configured. Use up()/down()/ready() for stack-only lifecycles."
            )
        container_name = target.container_name

        error_msg = self._ensure_target_started(target, container_name)
        if error_msg is not None:
            return r[m.Tests.ContainerInfo].fail(error_msg)
        return self._ensure_target_ready(target, container_name)

    def _ensure_target_started(
        self, target: m.Tests.ContainerConfig, container_name: str
    ) -> str | None:
        """Start or recreate the configured target when required."""
        if target.force_recreate or self.container_dirty(container_name):
            compose_result = self.compose_up(
                str(target.compose_file),
                service=target.service or None,
                force_recreate=True,
            )
            if compose_result.failure:
                return compose_result.error or "Failed to recreate Docker target"
            _ = self.mark_container_clean(container_name)
            return None

        status_result = self.fetch_container_status(container_name)
        container_running = status_result.success and (
            status_result.value.status == c.Tests.ContainerStatus.RUNNING
        )
        if container_running:
            return None
        start_result = self.start_existing_container(container_name)
        if start_result.success:
            return None
        compose_result = self.compose_up(
            str(target.compose_file), service=target.service or None
        )
        if compose_result.failure:
            return compose_result.error or "Failed to start Docker target"
        return None

    def _ensure_target_ready(
        self, target: m.Tests.ContainerConfig, container_name: str
    ) -> p.Result[m.Tests.ContainerInfo]:
        """Fetch target info and run configured readiness checks."""
        container_info_result = self.fetch_container_info(container_name)
        if container_info_result.failure:
            return container_info_result
        if target.port is None:
            return container_info_result

        ready_port = self._resolve_readiness_port(target, container_info_result.value)
        if ready_port is None:
            return r[m.Tests.ContainerInfo].fail(
                f"Docker target {target.container_name} has no resolved host port for readiness check"
            )
        ready_result = self.wait_for_port_ready(
            target.host, ready_port, max_wait=target.startup_timeout
        )
        if ready_result.failure:
            return r[m.Tests.ContainerInfo].fail(
                ready_result.error or "Docker target readiness check failed"
            )
        if not ready_result.value:
            return r[m.Tests.ContainerInfo].fail(
                f"Container {target.container_name} did not become ready on {target.host}:{ready_port}"
            )
        return container_info_result


tk = FlextTestsDocker


__all__: list[str] = ["FlextTestsDocker", "tk"]
