"""Structural documentation contracts for exact model transport."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Protocol, runtime_checkable

from flext_cli import p as cli_p


# NOTE (multi-agent, flext-wkii.17.23 / agent: uv_overlay_owner): protocols mirror
# source models structurally; no concrete m type crosses a public interface.
@runtime_checkable
class FlextInfraProtocolsDocs(Protocol):
    """Documentation model protocols exposed through ``p.Infra``."""

    @runtime_checkable
    class DocsRepositoryRef(Protocol):
        """Repository catalog fields consumed by documentation."""

        @property
        def name(self) -> str: ...

        @property
        def distribution(self) -> str: ...

        @property
        def url(self) -> str: ...

        @property
        def provider(self) -> str: ...

    @runtime_checkable
    class DocsProviderSpec(Protocol):
        """Git provider fields consumed by documentation."""

        @property
        def name(self) -> str: ...

        @property
        def organization(self) -> str: ...

        @property
        def base_url(self) -> str: ...

        @property
        def branch(self) -> str: ...

    @runtime_checkable
    class DocsExportBinding(Protocol):
        """Public export binding fields."""

        @property
        def export_name(self) -> str: ...

        @property
        def module_name(self) -> str: ...

    @runtime_checkable
    class DocsPublicContract(Protocol):
        """Exact source objects and derived public documentation facts."""

        @property
        def metadata(self) -> cli_p.ProjectMetadata: ...

        @property
        def repository(self) -> FlextInfraProtocolsDocs.DocsRepositoryRef | None: ...

        @property
        def provider(self) -> FlextInfraProtocolsDocs.DocsProviderSpec | None: ...

        @property
        def package_name(self) -> str: ...

        @property
        def doc_summary(self) -> str: ...

        @property
        def site_title(self) -> str: ...

        @property
        def site_url(self) -> str: ...

        @property
        def repo_url(self) -> str: ...

        @property
        def exports(self) -> tuple[str, ...]: ...

        @property
        def aliases(self) -> tuple[str, ...]: ...

        @property
        def facades(self) -> tuple[str, ...]: ...

        @property
        def module_exports(self) -> tuple[str, ...]: ...

        @property
        def public_symbols(self) -> tuple[str, ...]: ...

        @property
        def export_bindings(
            self,
        ) -> tuple[FlextInfraProtocolsDocs.DocsExportBinding, ...]: ...

        @property
        def modules(self) -> tuple[str, ...]: ...

        @property
        def source_paths(self) -> tuple[str, ...]: ...

    @runtime_checkable
    class MkDocsAnyCallable(Protocol):
        """Generic lazily-loaded MkDocs callable without loose top types."""

        def __call__(
            self, *args: cli_p.AttributeProbe, **kwargs: cli_p.AttributeProbe
        ) -> cli_p.AttributeProbe: ...

    @runtime_checkable
    class MkDocsLoadConfig(Protocol):
        """Contract for ``mkdocs.config.load_config``."""

        def __call__(
            self, *, config_file_path: str, site_dir: str
        ) -> MutableMapping[str, cli_p.AttributeProbe]: ...

    @runtime_checkable
    class MkDocsBuild(Protocol):
        """Contract for ``mkdocs.commands.build.build``."""

        def __call__(
            self,
            config: MutableMapping[str, cli_p.AttributeProbe],
            *,
            dirty: bool = False,
        ) -> cli_p.AttributeProbe: ...

    @runtime_checkable
    class MkDocsServe(Protocol):
        """Contract for ``mkdocs.commands.serve.serve``."""

        def __call__(
            self,
            *,
            config_file: str,
            livereload: bool,
            dev_addr: str,
            strict: bool = False,
        ) -> cli_p.AttributeProbe: ...


__all__: list[str] = ["FlextInfraProtocolsDocs"]
