"""Protocol definitions for FLEXT tests.

Provides FlextTestsProtocols, extending FlextProtocols with test-specific protocol
definitions for Docker operations, container management, and test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

from flext_core import FlextProtocols, T, r
from pydantic import BaseModel

from flext_tests.typings import t


class FlextTestsProtocols(FlextProtocols):
    """Protocol definitions for FLEXT tests - extends FlextProtocols.

    Architecture: Extends FlextProtocols with test-specific protocol definitions.
    All base protocols from FlextProtocols are available through inheritance pattern.
    Protocols cannot import models - only other protocols and types.
    """

    class Tests:
        """Test-specific protocol definitions."""

        class Docker:
            """Docker-specific protocol definitions."""

            @runtime_checkable
            class Container(Protocol):
                """Protocol for Docker container objects."""

                @property
                def id(self) -> str:
                    """Container ID."""
                    ...

                @property
                def name(self) -> str:
                    """Container name."""
                    ...

                @property
                def status(self) -> str:
                    """Container status."""
                    ...

                @property
                def image(self) -> str:
                    """Container image."""
                    ...

                def start(self) -> None:
                    """Start the container."""
                    ...

                def stop(self) -> None:
                    """Stop the container."""
                    ...

                def remove(
                    self,
                    *,
                    force: bool = False,
                    **kwargs: str | int | bool | None,
                ) -> None:
                    """Remove the container.

                    Args:
                        force: Force removal of running container
                        **kwargs: Additional removal options (labels, volumes, etc.)

                    """
                    ...

            @runtime_checkable
            class DockerClient(Protocol):
                """Protocol for Docker client operations."""

                def containers(
                    self,
                ) -> FlextTestsProtocols.Tests.Docker.ContainerCollection:
                    """Get container collection."""
                    ...

                def images(self) -> FlextTestsProtocols.Tests.Docker.ImageCollection:
                    """Get image collection."""
                    ...

                def networks(
                    self,
                ) -> FlextTestsProtocols.Tests.Docker.NetworkCollection:
                    """Get network collection."""
                    ...

                def volumes(
                    self,
                ) -> FlextTestsProtocols.Tests.Docker.VolumeCollection:
                    """Get volume collection."""
                    ...

            @runtime_checkable
            class ContainerCollection(Protocol):
                """Protocol for container collection operations."""

                def get(
                    self,
                    container_id: str,
                ) -> FlextTestsProtocols.Tests.Docker.Container:
                    """Get container by ID."""
                    ...

                def list(
                    self,
                    *,
                    show_all: bool = False,
                    filters: Mapping[str, str | Sequence[str]] | None = None,
                    **kwargs: str | int | bool | None,
                ) -> list[FlextTestsProtocols.Tests.Docker.Container]:
                    """List containers with filters.

                    Args:
                        show_all: Show all containers (including stopped)
                        filters: Filter mapping (e.g., {"status": ["running"]})
                        **kwargs: Additional filter options

                    """
                    ...

                def run(
                    self,
                    image: str,
                    *,
                    name: str | None = None,
                    detach: bool = True,
                    ports: Mapping[str, str | int] | None = None,
                    environment: Sequence[str] | None = None,
                    **kwargs: str | int | bool | None,
                ) -> FlextTestsProtocols.Tests.Docker.Container:
                    """Run a new container.

                    Args:
                        image: Container image name
                        name: Container name
                        detach: Run in detached mode
                        ports: Port mappings
                        environment: Environment variables
                        **kwargs: Additional container options

                    """
                    ...

            @runtime_checkable
            class ImageCollection(Protocol):
                """Protocol for image collection operations."""

                def get(self, image_id: str) -> FlextTestsProtocols.Tests.Docker.Image:
                    """Get image by ID."""
                    ...

                def list(
                    self,
                    *,
                    show_all: bool = False,
                    filters: Mapping[str, str | Sequence[str]] | None = None,
                    **kwargs: str | int | bool | None,
                ) -> list[FlextTestsProtocols.Tests.Docker.Image]:
                    """List images with filters.

                    Args:
                        show_all: Show all images (including intermediate)
                        filters: Filter mapping
                        **kwargs: Additional filter options

                    """
                    ...

                def build(
                    self,
                    path: str,
                    *,
                    tag: str | None = None,
                    build_args: Mapping[str, str] | None = None,
                    **kwargs: str | int | bool | None,
                ) -> FlextTestsProtocols.Tests.Docker.Image:
                    """Build an image.

                    Args:
                        path: Build context path
                        tag: Image tag
                        build_args: Build arguments
                        **kwargs: Additional build options

                    """
                    ...

            @runtime_checkable
            class Image(Protocol):
                """Protocol for Docker image objects."""

                @property
                def id(self) -> str:
                    """Image ID."""
                    ...

                @property
                def tags(self) -> list[str]:
                    """Image tags."""
                    ...

            @runtime_checkable
            class NetworkCollection(Protocol):
                """Protocol for network collection operations."""

                def get(
                    self,
                    network_id: str,
                ) -> FlextTestsProtocols.Tests.Docker.Network:
                    """Get network by ID."""
                    ...

                def list(
                    self,
                    *,
                    filters: Mapping[str, str | Sequence[str]] | None = None,
                    **kwargs: str | int | bool | None,
                ) -> list[FlextTestsProtocols.Tests.Docker.Network]:
                    """List networks with filters.

                    Args:
                        filters: Filter mapping
                        **kwargs: Additional filter options

                    """
                    ...

                def create(
                    self,
                    name: str,
                    *,
                    driver: str = "bridge",
                    ipam: Mapping[str, str | Sequence[str]] | None = None,
                    **kwargs: str | int | bool | None,
                ) -> FlextTestsProtocols.Tests.Docker.Network:
                    """Create a network.

                    Args:
                        name: Network name
                        driver: Network driver
                        ipam: IPAM configuration
                        **kwargs: Additional network options

                    """
                    ...

            @runtime_checkable
            class Network(Protocol):
                """Protocol for Docker network objects."""

                @property
                def id(self) -> str:
                    """Network ID."""
                    ...

                @property
                def name(self) -> str:
                    """Network name."""
                    ...

            @runtime_checkable
            class VolumeCollection(Protocol):
                """Protocol for volume collection operations."""

                def get(
                    self,
                    volume_id: str,
                ) -> FlextTestsProtocols.Tests.Docker.Volume:
                    """Get volume by ID."""
                    ...

                def list(
                    self,
                    *,
                    filters: Mapping[str, str | Sequence[str]] | None = None,
                    **kwargs: str | int | bool | None,
                ) -> list[FlextTestsProtocols.Tests.Docker.Volume]:
                    """List volumes with filters.

                    Args:
                        filters: Filter mapping
                        **kwargs: Additional filter options

                    """
                    ...

                def create(
                    self,
                    name: str,
                    *,
                    driver: str | None = None,
                    driver_opts: Mapping[str, str] | None = None,
                    **kwargs: str | int | bool | None,
                ) -> FlextTestsProtocols.Tests.Docker.Volume:
                    """Create a volume.

                    Args:
                        name: Volume name
                        driver: Volume driver
                        driver_opts: Driver options
                        **kwargs: Additional volume options

                    """
                    ...

            @runtime_checkable
            class Volume(Protocol):
                """Protocol for Docker volume objects."""

                @property
                def id(self) -> str:
                    """Volume ID."""
                    ...

                @property
                def name(self) -> str:
                    """Volume name."""
                    ...

            @runtime_checkable
            class ComposeClient(Protocol):
                """Protocol for docker-compose operations.

                Compatible with python-on-whales DockerClient.
                Uses structural typing - any object with compose/client_config.
                """

                compose: t.ConfigMapValue
                """Compose API access (python-on-whales style)."""

                client_config: Mapping[str, t.ConfigMapValue]
                """Client configuration (python-on-whales style)."""

                def up(
                    self,
                    services: Sequence[str] | None = None,
                    *,
                    detach: bool = True,
                    build: bool = False,
                    **kwargs: str | int | bool | None,
                ) -> None:
                    """Start compose services.

                    Args:
                        services: Specific services to start
                        detach: Run in detached mode
                        build: Build images before starting
                        **kwargs: Additional compose options

                    """
                    ...

                def down(
                    self,
                    *,
                    volumes: bool = False,
                    remove_orphans: bool = False,
                    timeout: int | None = None,
                    **kwargs: str | int | bool | None,
                ) -> None:
                    """Stop compose services.

                    Args:
                        volumes: Remove volumes
                        remove_orphans: Remove orphan containers
                        timeout: Stop timeout in seconds
                        **kwargs: Additional compose options

                    """
                    ...

                def restart(
                    self,
                    services: Sequence[str] | None = None,
                    *,
                    timeout: int | None = None,
                    **kwargs: str | int | bool | None,
                ) -> None:
                    """Restart compose services.

                    Args:
                        services: Specific services to restart
                        timeout: Restart timeout in seconds
                        **kwargs: Additional compose options

                    """
                    ...

        class Factory:
            """Factory-specific protocol definitions for test data generation.

            Provides structural typing for factory operations following FLEXT patterns.
            All protocols use types from t.Tests.Factory.* for consistency.
            """

            @runtime_checkable
            class ModelFactory(Protocol):
                """Protocol for model factory operations.

                Compatible with FlextTestsFactories.model() method.
                Uses structural typing - any callable matching this signature.
                All parameters validated via ModelFactoryParams with u.Model.from_kwargs().
                """

                def __call__(
                    self,
                    kind: str = ...,
                    # All parameters via kwargs - validated by ModelFactoryParams
                    **kwargs: t.Tests.PayloadValue,
                ) -> (
                    t.Tests.PayloadValue
                    | list[t.Tests.PayloadValue]
                    | Mapping[str, t.Tests.PayloadValue]
                    | r[t.Tests.PayloadValue]
                    | r[list[t.Tests.PayloadValue]]
                    | r[Mapping[str, t.Tests.PayloadValue]]
                ):
                    """Create model instance(s) with optional transformations.

                    Args:
                        kind: Model type to create
                        **kwargs: All parameters validated by ModelFactoryParams:
                            - count: Number of instances (returns list if > 1)
                            - as_dict: Return as dict with ID keys
                            - as_result: Wrap in FlextResult
                            - as_mapping: Map to custom keys
                            - factory: Custom factory callable
                            - transform: Post-transform function
                            - validate: Validation predicate
                            - model_id, name, email, active, etc.: Model-specific fields
                            - **overrides: Override any field directly

                    Returns:
                        Model instance, list, dict, or FlextResult wrapping any

                    """
                    ...

            @runtime_checkable
            class ResultFactory(Protocol):
                """Protocol for result factory operations.

                Compatible with FlextTestsFactories.res() method.
                Uses structural typing for FlextResult creation.
                All parameters validated via ResultFactoryParams with u.Model.from_kwargs().
                """

                def __call__[TValue](
                    self,
                    kind: str = "ok",
                    value: TValue | None = None,
                    # All parameters via kwargs - validated by ResultFactoryParams
                    **kwargs: t.Tests.PayloadValue,
                ) -> (
                    FlextProtocols.Result[TValue] | list[FlextProtocols.Result[TValue]]
                ):
                    """Create FlextResult instance(s) with full customization.

                    Args:
                        kind: Result type ('ok', 'fail', 'from_value')
                        value: Value for success results
                        **kwargs: All parameters validated by ResultFactoryParams:
                            - count: Number of results to create
                            - values: Explicit value list for batch
                            - errors: Error messages for failures
                            - mix_pattern: Success/failure pattern
                            - error: Error message for failures
                            - error_code: Optional error code
                            - error_on_none: Error when value is None
                            - transform: Transform function for values

                    Returns:
                        FlextResult or list of FlextResult instances

                    """
                    ...

            @runtime_checkable
            class CollectionFactory(Protocol):
                """Protocol for collection factory operations.

                Compatible with FlextTestsFactories.list() and dict() methods.
                Uses structural typing for collection creation.
                All parameters validated via ListFactoryParams/DictFactoryParams with u.Model.from_kwargs().
                """

                def list[T](
                    self,
                    source: t.Tests.PayloadValue = "user",
                    # All parameters via kwargs - validated by ListFactoryParams
                    **kwargs: t.Tests.PayloadValue,
                ) -> list[T] | r[list[T]]:
                    """Create typed list from source.

                    Args:
                        source: Source for items (Sequence, Callable, or ModelKind)
                        **kwargs: All parameters validated by ListFactoryParams:
                            - count: Number of items to create
                            - as_result: Wrap in FlextResult
                            - unique: Ensure uniqueness
                            - transform: Transform each item
                            - filter: Filter predicate

                    Returns:
                        List of items or FlextResult wrapping list

                    """
                    ...

                def dict[K, V](
                    self,
                    source: (Mapping[K, V] | t.Tests.PayloadValue) = "user",
                    # All parameters via kwargs - validated by DictFactoryParams
                    **kwargs: t.Tests.PayloadValue,
                ) -> Mapping[K, V] | r[Mapping[K, V]]:
                    """Create typed dict from source.

                    Args:
                        source: Source for items (Mapping, Callable, or ModelKind)
                        **kwargs: All parameters validated by DictFactoryParams:
                            - count: Number of items to create
                            - key_factory: Factory for keys
                            - value_factory: Factory for values
                            - as_result: Wrap in FlextResult
                            - merge_with: Additional mapping to merge

                    Returns:
                        Dict of items or FlextResult wrapping dict

                    """
                    ...

            @runtime_checkable
            class GenericFactory(Protocol):
                """Protocol for generic type factory operations.

                Compatible with FlextTestsFactories.generic() method.
                Uses structural typing for generic type instantiation.
                All parameters validated via GenericFactoryParams with u.Model.from_kwargs().
                """

                def __call__[T](
                    self,
                    type_: type[T],
                    # All parameters via kwargs - validated by GenericFactoryParams
                    **kwargs: t.Tests.PayloadValue,
                ) -> T | list[T] | r[T] | r[list[T]]:
                    """Create instance(s) of any type with full type safety.

                    Args:
                        type_: Type class to instantiate
                        **kwargs: All parameters validated by GenericFactoryParams:
                            - args: Positional arguments for constructor
                            - kwargs: Keyword arguments for constructor
                            - count: Number of instances
                            - as_result: Wrap in FlextResult
                            - validate: Validation predicate

                    Returns:
                        Instance, list, or FlextResult wrapping any

                    """
                    ...

        class Matcher:
            """Matcher protocol definitions for test assertions (tm.* methods).

            Provides structural typing for matcher operations without requiring
            inheritance. All protocols are runtime-checkable.
            """

            @runtime_checkable
            class Assertion(Protocol):
                """Protocol for assertion operations.

                Structural typing for objects that can perform assertions.
                Used for validation and testing operations.
                """

                def assert_ok(self, result: T) -> T:
                    """Assert result is success and return value."""
                    ...

                def assert_fail(self, result: r[t.ConfigMapValue]) -> str:
                    """Assert result is failure and return error."""
                    ...

                def assert_that(
                    self,
                    value: T,
                    **kwargs: T,
                ) -> None:
                    """Assert value satisfies conditions."""
                    ...

            @runtime_checkable
            class DeepMatcher(Protocol):
                """Protocol for deep structural matching operations.

                Structural typing for objects that can perform deep matching.
                """

                def match(
                    self,
                    obj: T,
                    spec: Mapping[
                        str,
                        t.Tests.PayloadValue | Callable[[t.Tests.PayloadValue], bool],
                    ],
                    *,
                    path_sep: str = ".",
                ) -> T:
                    """Match object against deep specification.

                    Args:
                        obj: Object to match against
                        spec: Mapping of path -> expected value or predicate
                        path_sep: Path separator (default: ".")

                    Returns:
                        Match result with path, expected, actual, matched, reason

                    """
                    ...

            @runtime_checkable
            class LengthValidator(Protocol):
                """Protocol for length validation operations.

                Structural typing for objects that can validate lengths.
                """

                def validate(
                    self,
                    value: t.ConfigMapValue,
                    spec: int | tuple[int, int],
                ) -> bool:
                    """Validate length against spec.

                    Args:
                        value: Value to check length of
                        spec: Length spec (exact int or (min, max) tuple)

                    Returns:
                        True if length matches spec

                    """
                    ...

            @runtime_checkable
            class ChainBuilder(Protocol):
                """Protocol for fluent chain builder operations.

                Structural typing for objects that support chained assertions.
                """

                def ok(self, msg: str | None = None) -> Self:
                    """Assert result is success."""
                    ...

                def fail(
                    self,
                    error: str | None = None,
                    msg: str | None = None,
                ) -> Self:
                    """Assert result is failure."""
                    ...

                def eq(
                    self,
                    expected: t.ConfigMapValue,
                    msg: str | None = None,
                ) -> Self:
                    """Assert value equals expected."""
                    ...

                def has(self, item: t.ConfigMapValue, msg: str | None = None) -> Self:
                    """Assert value/error contains item."""
                    ...

                def len(self, expected: int, msg: str | None = None) -> Self:
                    """Assert value has expected length."""
                    ...

                def done(self) -> Self:
                    """Finish chain and return value (for success)."""
                    ...

                def err(self) -> str:
                    """Finish chain and return error (for failure)."""
                    ...

            @runtime_checkable
            class ScopeManager(Protocol):
                """Protocol for test scope management operations.

                Structural typing for objects that can manage test execution scopes.
                """

                def scope(
                    self,
                    **kwargs: T,
                ) -> T:
                    """Enter test execution scope with **kwargs pattern.

                    Args:
                        **kwargs: Scope parameters (config, container, context,
                            cleanup, env, cwd) - validated via ScopeParams model

                    Returns:
                        Scope object with config, container, context

                    """
                    ...

                def exit_scope(self, scope: t.ConfigMapValue) -> None:
                    """Exit test execution scope and cleanup.

                    Args:
                        scope: Scope object to exit

                    """
                    ...

        class Files:
            """File-specific protocol definitions for test file operations (tf).

            Provides structural typing for file operations following FLEXT patterns.
            All protocols use types from t.Tests.Files.* and protocols from
            FlextProtocols.Result for consistency.
            """

            @runtime_checkable
            class FileOperation(Protocol):
                """Protocol for file operation methods.

                Compatible with FlextTestsFiles core methods (create, read, compare, info, batch).
                Uses structural typing - any callable matching these signatures.
                Follows FLEXT patterns using FlextProtocols.Result for return types.
                """

                def create(
                    self,
                    content: t.Tests.PayloadValue,
                    name: str = ...,
                    directory: str | None = ...,
                    **kwargs: str | float | bool | None,
                ) -> FlextProtocols.Result[Path]:
                    """Create file with auto-detection.

                    Returns:
                        Result protocol wrapping Path on success.

                    """
                    ...

                def read(
                    self,
                    path: Path | str,
                    *,
                    model_cls: type[BaseModel] | None = ...,
                    **kwargs: str | float | bool | None,
                ) -> FlextProtocols.Result[t.Tests.PayloadValue]:
                    """Read file with optional model deserialization.

                    Returns:
                        Result protocol wrapping content or model instance.

                    """
                    ...

                def compare(
                    self,
                    file1: Path | str,
                    file2: Path | str,
                    *,
                    mode: str = ...,
                    **kwargs: str | float | bool | None,
                ) -> FlextProtocols.Result[bool]:
                    """Compare two files.

                    Returns:
                        Result protocol wrapping bool (True if match).

                    """
                    ...

                def info(
                    self,
                    path: Path | str,
                    *,
                    compute_hash: bool = ...,
                    detect_fmt: bool = ...,
                    **kwargs: str | float | bool | None,
                ) -> FlextProtocols.Result[t.Tests.PayloadValue]:
                    """Get comprehensive file information.

                    Returns:
                        Result protocol wrapping FileInfo model.

                    """
                    ...

                def batch(
                    self,
                    files: Sequence[tuple[str, str]] | Mapping[str, str],
                    *,
                    directory: Path | str | None = ...,
                    operation: str = ...,
                    model: type[BaseModel] | None = ...,
                    on_error: str = ...,
                    **kwargs: str | float | bool | None,
                ) -> FlextProtocols.Result[t.Tests.PayloadValue]:
                    """Batch file operations.

                    Returns:
                        Result protocol wrapping BatchResult model.

                    """
                    ...

        class Builders:
            """Builder protocol definitions for test data construction (tb).

            Provides structural typing for builder operations without requiring
            inheritance. All protocols are runtime-checkable.
            """

            @runtime_checkable
            class Builder(Protocol):
                """Protocol for builder operations.

                Structural typing for objects that can build test data.
                Used for fluent interface pattern.
                """

                def add(
                    self,
                    key: str,
                    value: t.Tests.PayloadValue | None = ...,
                    **kwargs: t.Tests.PayloadValue,
                ) -> Self:
                    """Add data to builder.

                    Args:
                        key: Key to store data under
                        value: Direct value to store
                        **kwargs: Additional parameters (factory, count, etc.)

                    Returns:
                        Self for method chaining

                    """
                    ...

                def set(
                    self,
                    path: str,
                    value: t.Tests.PayloadValue | None = ...,
                    *,
                    create_parents: bool = ...,
                    **kwargs: t.Tests.PayloadValue,
                ) -> Self:
                    """Set value at nested path.

                    Args:
                        path: Dot-separated path
                        value: Value to set
                        create_parents: Whether to create intermediate dicts
                        **kwargs: Additional values to set as mapping

                    Returns:
                        Self for method chaining

                    """
                    ...

                def get[T](
                    self,
                    path: str,
                    default: T | None = ...,
                    *,
                    as_type: type[T] | None = ...,
                ) -> T | None:
                    """Get value from path.

                    Args:
                        path: Dot-separated path
                        default: Default value if not found
                        as_type: Type to cast result to

                    Returns:
                        Value at path or default

                    """
                    ...

                def build(
                    self,
                    **kwargs: t.Tests.PayloadValue,
                ) -> (
                    Mapping[str, t.Tests.PayloadValue]
                    | BaseModel
                    | list[tuple[str, t.Tests.PayloadValue]]
                    | list[str]
                    | list[t.Tests.PayloadValue]
                    | t.Tests.PayloadValue
                ):
                    """Build the dataset with output type control.

                    Args:
                        **kwargs: Build parameters (as_model, as_list, etc.)

                    Returns:
                        Built dataset in requested format

                    """
                    ...

                def to_result[T](
                    self,
                    **kwargs: t.Tests.PayloadValue,
                ) -> (
                    r[T]
                    | r[Mapping[str, t.Tests.PayloadValue]]
                    | r[BaseModel]
                    | r[list[T]]
                    | r[Mapping[str, T]]
                    | T
                ):
                    """Build data wrapped in FlextResult.

                    Args:
                        **kwargs: Result parameters (as_model, error, unwrap, etc.)

                    Returns:
                        FlextResult containing built data or unwrapped value

                    """
                    ...

                def copy_builder(self) -> Self:
                    """Create independent copy of builder state.

                    Returns:
                        New builder instance with copied data

                    """
                    ...

                def fork(self, **updates: t.Tests.PayloadValue) -> Self:
                    """Copy and immediately add updates.

                    Args:
                        **updates: Key-value pairs to add to copied builder

                    Returns:
                        New builder instance with copied data and updates

                    """
                    ...

                def merge_from(
                    self,
                    other: FlextTestsProtocols.Tests.Builders.Builder,
                    *,
                    strategy: str = ...,
                    exclude_keys: builtins.set[str] | None = ...,
                ) -> Self:
                    """Merge data from another builder.

                    Args:
                        other: Another builder to merge from
                        strategy: Merge strategy
                        exclude_keys: Set of keys to exclude from merge

                    Returns:
                        Self for method chaining

                    """
                    ...

                def batch(
                    self,
                    key: str,
                    scenarios: Sequence[tuple[str, t.Tests.PayloadValue]],
                    **kwargs: t.Tests.PayloadValue,
                ) -> Self:
                    """Build batch of test scenarios.

                    Args:
                        key: Key to store batch under
                        scenarios: Sequence of (scenario_id, data) tuples
                        **kwargs: Additional parameters (as_results, with_failures)

                    Returns:
                        Self for method chaining

                    """
                    ...

                def scenarios(
                    self,
                    *cases: tuple[str, Mapping[str, t.Tests.PayloadValue]],
                ) -> list[tuple[str, Mapping[str, t.Tests.PayloadValue]]]:
                    """Build pytest.mark.parametrize compatible scenarios.

                    Args:
                        *cases: Variable number of (test_id, data) tuples

                    Returns:
                        List of parametrized test cases

                    """
                    ...

                def reset(self) -> Self:
                    """Reset builder state.

                    Returns:
                        Self for method chaining

                    """
                    ...

        class Support:
            """Support protocols for type system compatibility."""

            @runtime_checkable
            class SupportsLessThan(Protocol):
                """Protocol for types that support less-than comparison.

                Required by sorted() key functions - the return type must
                be comparable. This protocol ensures type safety for key
                functions used with sorted().

                Example:
                    def get_id(obj: t.ConfigMapValue) -> int:
                        return obj.id  # int supports __lt__

                    sorted(items, key=get_id)  # OK - int satisfies SupportsLessThan

                """

                def __lt__(self, other: object, /) -> bool:
                    """Less-than comparison operator."""
                    ...


p = FlextTestsProtocols

__all__ = ["FlextTestsProtocols", "p"]
