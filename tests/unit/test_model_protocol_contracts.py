"""Exercise real models through their structural consumer contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests import m, p, tm, u


class TestsModelProtocolContracts:
    """Keep callable assignment and nested payload validation model-owned."""

    @staticmethod
    def test_factory_contract_preserves_callable_assignment() -> None:
        """The model satisfies the writable port without synthetic properties."""
        registration: p.FactoryRegistration = m.FactoryRegistration(
            name="protocol-factory", factory=lambda: "initial"
        )
        tm.that(registration.factory(), eq="initial")
        registration.factory = lambda: "replacement"
        tm.that(registration.factory(), eq="replacement")

    @staticmethod
    def test_resource_contract_preserves_callable_assignment() -> None:
        """A resource factory remains writable through its protocol."""
        registration: p.ResourceRegistration = m.ResourceRegistration(
            name="protocol-resource", factory=lambda: "initial"
        )
        tm.that(registration.factory(), eq="initial")
        registration.factory = lambda: "replacement"
        tm.that(registration.factory(), eq="replacement")

    @staticmethod
    def test_root_contract_preserves_callable_assignment() -> None:
        """RootModel retains its native constructor and writable callable root."""
        validator: p.ValidatorCallable = m.ValidatorCallable(root=str)
        tm.that(validator("initial"), eq="initial")
        validator.root = lambda value: str(value).upper()
        tm.that(validator("replacement"), eq="REPLACEMENT")

    @staticmethod
    def test_nested_payload_retains_native_models_and_serialization() -> None:
        """A protocol consumer receives validated concrete nested models."""
        project: p.Project = m.Project.model_validate_json(
            '{"name":"protocol-project","version":"1.0",'
            '"authors":[{"name":"Ada"}],'
            '"urls":{"Homepage":"https://example.com"}}'
        )
        author: p.ProjectAuthor = project.authors[0]
        tm.that(author.name, eq="Ada")
        tm.that(project.urls.homepage, eq="https://example.com")
        tm.that(m.Project.model_validate_json(project.model_dump_json()), eq=project)
        tm.that(
            m.Project.model_json_schema()["properties"]["authors"]["items"]["$ref"],
            eq="#/$defs/ProjectAuthor",
        )

    @staticmethod
    def test_nested_payload_rejects_invalid_author() -> None:
        """The concrete field schema rejects invalid nested external input."""
        with u.Tests.raises(m.ValidationError) as raised:
            m.Project.model_validate_json(
                '{"name":"protocol-project","version":"1.0","authors":[{"name":7}]}'
            )
        tm.that(raised.value.errors()[0]["loc"], eq=("authors", 0, "name"))
