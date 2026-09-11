"""Kubernetes (kind) cluster control facade for FLEXT test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Self, override

from flext_tests import c, m, p, r, t, u
from flext_tests.docker import FlextTestsDocker


class FlextTestsKube(FlextTestsDocker):
    """Manage a kind Kubernetes cluster for FLEXT tests via docker compose.

    Specializes :class:`FlextTestsDocker` for the ``flext-kind-test`` shared
    container entry: brings the consumer-owned
    ``docker/docker-compose.kubernetes.yml`` stack up, waits for the
    apiserver port, and asserts node readiness through ``kubectl``.
    """

    KIND_CONTAINER_NAME: ClassVar[str] = "flext-kind-test"

    kubectl_service: Annotated[
        str,
        u.Field(
            min_length=1,
            description=(
                "Compose service exposing kubectl against the kind kubeconfig."
            ),
        ),
    ] = "kubectl"

    @classmethod
    def kind(
        cls, *, workspace_root: Path | None = None, worker_id: str | None = None
    ) -> Self:
        """Build a DSL-configured service for the shared kind cluster."""
        resolved_root = workspace_root or Path.cwd()
        return cls(
            workspace_root=resolved_root,
            worker_id=worker_id or "master",
            target_config=cls._resolve_shared_target_config(
                cls.KIND_CONTAINER_NAME, resolved_root
            ),
        )

    def cluster_up(self) -> p.Result[str]:
        """Start the kind stack and wait for the apiserver to accept TCP."""
        target = self.target_config
        if target is None:
            return r[str].fail(
                "Kubernetes target not configured. Use tkube.kind(...) first."
            )
        if target.compose_file is None:
            return r[str].fail("Kubernetes target has no compose file configured.")
        up_result = self.compose_up(
            str(target.compose_file),
            service=target.service or None,
            force_recreate=target.force_recreate,
        )
        if up_result.failure:
            return r[str].fail_op("Kind cluster start", up_result.error)
        if target.port is None:
            return r[str].ok("Kind cluster started (no readiness port configured)")
        ready = self.wait_for_port_ready(
            target.host, target.port, max_wait=target.startup_timeout
        )
        if ready.failure:
            return r[str].fail_op("Kind apiserver readiness", ready.error)
        if not ready.value:
            return r[str].fail(
                f"Kind apiserver did not become ready on {target.host}:{target.port}"
            )
        return r[str].ok("Kind cluster started and apiserver is reachable")

    def cluster_down(self) -> p.Result[str]:
        """Tear down the kind stack via compose down."""
        target = self.target_config
        if target is None:
            return r[str].fail(
                "Kubernetes target not configured. Use tkube.kind(...) first."
            )
        if target.compose_file is None:
            return r[str].fail("Kubernetes target has no compose file configured.")
        return self.compose_down(str(target.compose_file))

    def nodes_ready(self) -> p.Result[bool]:
        """Run ``kubectl get nodes`` and confirm every node reports Ready."""
        target = self.target_config
        if target is None or target.compose_file is None:
            return r[bool].fail(
                "Kubernetes target not configured. Use tkube.kind(...) first."
            )
        try:
            output = self._run_kubectl(["get", "nodes", "--no-headers"])
        except self._compose_exception_types() as exc:
            return r[bool].fail_op("kubectl get nodes", exc)
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return r[bool].fail("kubectl get nodes returned no nodes")
        not_ready = [line for line in lines if " Ready" not in f" {line}"]
        if not_ready:
            return r[bool].fail(f"Kind nodes not Ready: {not_ready}")
        return r[bool].ok(value=True)

    def _run_kubectl(self, args: t.StrSequence) -> str:
        """Exec kubectl through the compose sidecar against the kind cluster."""
        target = self.target_config
        if target is None or target.compose_file is None:
            msg = "Kubernetes target not configured"
            raise ValueError(msg)
        original_files = self.docker.client_config.compose_files
        try:
            self.docker.client_config.compose_files = [str(target.compose_file)]
            output = self.docker.compose.execute(
                self.kubectl_service, list(args), tty=False
            )
        finally:
            self.docker.client_config.compose_files = original_files
        return output or ""

    @override
    def execute(self) -> p.Result[m.Tests.ContainerInfo]:
        """Bring the kind cluster up, verify node readiness, and return info."""
        target = self.target_config
        if target is None:
            return r[m.Tests.ContainerInfo].fail(
                "Kubernetes target not configured. Use tkube.kind(...).execute()."
            )
        up_result = self.cluster_up()
        if up_result.failure:
            return r[m.Tests.ContainerInfo].fail(
                up_result.error or "Kind cluster start failed"
            )
        nodes_result = self.nodes_ready()
        if nodes_result.failure:
            return r[m.Tests.ContainerInfo].fail(
                nodes_result.error or "Kind nodes readiness check failed"
            )
        container_name = target.container_name
        if not container_name:
            return r[m.Tests.ContainerInfo].ok(
                m.Tests.ContainerInfo(
                    name=target.service or self.KIND_CONTAINER_NAME,
                    status=c.Tests.ContainerStatus.RUNNING,
                    ports={},
                    image="",
                )
            )
        return self.fetch_container_info(container_name)


tkube = FlextTestsKube


__all__: list[str] = ["FlextTestsKube", "tkube"]
