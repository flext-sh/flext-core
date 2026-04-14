"""Comprehensive coverage tests for FlextModels DDD patterns."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, override

import pytest
from pydantic import Field, ValidationError, field_validator

from flext_core import FlextModelsCqrs
from flext_tests import tm
from tests import m, t, u


class GetUserQuery(m.Query):
    """Query to get user."""


class ListAccountsQuery(m.Query):
    """Query to list accounts."""

    page: Annotated[int, Field()]
    limit: Annotated[int, Field()]


class SearchProductsQuery(m.Query):
    """Query to search products."""

    keyword: Annotated[str, Field()]
    category: Annotated[str | None, Field(default=None)] = None
    min_price: Annotated[float | None, Field(default=None)] = None
    max_price: Annotated[float | None, Field(default=None)] = None


class TestCoverageModels:
    def test_value_object_creation(self) -> None:
        class Money(m.Value):
            amount: float
            currency: str

        money = Money(amount=100.0, currency="USD")
        tm.that(math.isclose(money.amount, 100.0), eq=True)
        tm.that(money.currency, eq="USD")

    def test_value_object_immutability(self) -> None:
        class Point(m.Value):
            x: float
            y: float

        point = Point(x=1.0, y=2.0)
        with pytest.raises(ValidationError):
            setattr(point, "x", 3.0)

    def test_value_object_equality_by_value(self) -> None:
        class Color(m.Value):
            red: int
            green: int
            blue: int

        color1 = Color(red=255, green=0, blue=0)
        color2 = Color(red=255, green=0, blue=0)
        tm.that(color1, eq=color2)

    def test_value_object_validation(self) -> None:
        class Email(m.Value):
            address: str

            @field_validator("address")
            @classmethod
            def validate_email(cls, value: str) -> str:
                if "@" not in value:
                    error_msg = "Invalid email format"
                    raise ValueError(error_msg)
                return value.lower()

        email = Email(address="USER@EXAMPLE.COM")
        tm.that(email.address, eq="user@example.com")
        with pytest.raises(ValidationError):
            Email(address="notanemail")

    def test_value_object_hashable(self) -> None:
        class ISBN(m.Value):
            code: str

            def __hash__(self) -> int:
                return hash(self.code)

            @override
            def __eq__(self, other: t.ValueOrModel) -> bool:
                if not isinstance(other, ISBN):
                    return False
                return self.code == other.code

        isbn1 = ISBN(code="978-0-262-03384-8")
        isbn2 = ISBN(code="978-0-262-03384-8")
        isbn_set: set[ISBN] = {isbn1, isbn2}
        tm.that(len(isbn_set), eq=1)

    def test_entity_creation(self) -> None:
        class Person(m.Entity):
            name: str
            age: int

        person = Person(name="Alice", age=30, domain_events=[])
        tm.that(person.name, eq="Alice")
        tm.that(person.unique_id, none=False)
        tm.that(person.created_at, none=False)
        tm.that(person.updated_at, none=False)

    def test_entity_identity_tracking(self) -> None:
        class Account(m.Entity):
            name: str
            balance: float

        account1 = Account(name="Checking", balance=100.0, domain_events=[])
        account2 = Account(name="Checking", balance=100.0, domain_events=[])
        tm.that(account1.unique_id, ne=account2.unique_id)
        tm.that(account1, ne=account2)

    def test_entity_lifecycle_tracking(self) -> None:
        class Document(m.Entity):
            title: str

        doc = Document(title="Test Doc", domain_events=[])
        tm.that(doc.created_at, none=False)
        tm.that(doc.updated_at, none=False)
        assert doc.updated_at is not None
        tm.that(doc.created_at <= doc.updated_at, eq=True)

    def test_entity_validation(self) -> None:
        class User(m.Entity):
            email: str
            username: str

            @field_validator("username")
            @classmethod
            def validate_username(cls, value: str) -> str:
                if len(value) < 3:
                    error_msg = "Username must be at least 3 characters"
                    raise ValueError(error_msg)
                return value

        user = User(email="user@example.com", username="alice", domain_events=[])
        tm.that(user.username, eq="alice")
        with pytest.raises(ValidationError):
            User(email="user@example.com", username="ab", domain_events=[])

    def test_entity_model_dump_serialization(self) -> None:
        class Product(m.Entity):
            name: str
            price: float

        product = Product(name="Widget", price=19.99, domain_events=[])
        product_dict = product.model_dump()
        tm.that(product_dict, is_=dict)
        tm.that(product_dict["name"], eq="Widget")
        tm.that(math.isclose(product_dict["price"], 19.99), eq=True)
        tm.that(
            all(
                key in product_dict for key in ["unique_id", "created_at", "updated_at"]
            ),
            eq=True,
        )

    def test_aggregate_root_creation(self) -> None:
        class Order(m.AggregateRoot):
            order_number: str
            status: str

        order = Order(order_number="ORD-001", status="pending", domain_events=[])
        tm.that(order.order_number, eq="ORD-001")
        tm.that(order.unique_id, none=False)

    def test_aggregate_root_invariants(self) -> None:
        class Account(m.AggregateRoot):
            balance: float
            currency: str

        account = Account(balance=1000.0, currency="USD", domain_events=[])
        tm.that(account.balance, gte=0.0)

    def test_aggregate_root_lifecycle(self) -> None:
        class Project(m.AggregateRoot):
            name: str
            status: str

        project = Project(name="New Project", status="planning", domain_events=[])
        tm.that(project.status, eq="planning")
        tm.that(project.created_at, none=False)

    def test_command_creation(self) -> None:
        class CreateUserCommand(m.Command):
            email: str
            username: str

        cmd = CreateUserCommand(
            email="user@example.com",
            username="alice",
            command_id="cmd-test-1",
        )
        tm.that(cmd.email, eq="user@example.com")
        tm.that(cmd.command_id, none=False)

    def test_command_mutation_behavior(self) -> None:
        class UpdateProfileCommand(m.Command):
            name: str
            bio: str

        cmd = UpdateProfileCommand(
            name="Alice",
            bio="Developer",
            command_id="cmd-test-2",
        )
        original_name = cmd.name
        cmd.name = "Bob"
        tm.that(cmd.name, eq="Bob")
        tm.that(cmd.name, ne=original_name)

    def test_command_validation(self) -> None:
        class DepositCommand(m.Command):
            account_id: str
            amount: float

            @field_validator("amount")
            @classmethod
            def validate_amount(cls, value: float) -> float:
                if value <= 0:
                    error_msg = "Amount must be positive"
                    raise ValueError(error_msg)
                return value

        cmd = DepositCommand(
            account_id="ACC-001",
            amount=100.0,
            command_id="cmd-test-3",
        )
        tm.that(math.isclose(cmd.amount, 100.0), eq=True)
        with pytest.raises(ValidationError):
            DepositCommand(account_id="ACC-001", amount=-50.0, command_id="cmd-test-4")

    def test_query_creation(self) -> None:
        assert (
            u.resolve_nested_model_class(
                module_name=GetUserQuery.__module__,
                qualname=GetUserQuery.__qualname__,
                models_module_name="flext_core",
                attribute_name="Pagination",
                fallback=FlextModelsCqrs.Pagination,
            )
            is FlextModelsCqrs.Pagination
        )
        query = GetUserQuery(
            filters=t.Dict(root={"user_id": "USER-001"}),
            query_type="get_user",
            pagination=m.Pagination(),
            query_id="q-test-1",
        )
        tm.that(query.filters["user_id"], eq="USER-001")
        tm.that(query.query_id, none=False)
        tm.that(query.query_type, eq="get_user")

    def test_query_mutation_behavior(self) -> None:
        query = ListAccountsQuery(
            page=1,
            limit=10,
            filters=t.Dict(root={}),
            pagination=m.Pagination(),
            query_id="q-test-2",
        )
        original_page = query.page
        query.page = 2
        tm.that(query.page, eq=2)
        tm.that(query.page, ne=original_page)

    def test_query_with_filters(self) -> None:
        query = SearchProductsQuery(
            keyword="laptop",
            category="electronics",
            min_price=500.0,
            filters=t.Dict(root={}),
            pagination=m.Pagination(),
            query_id="q-test-3",
        )
        tm.that(query.keyword, eq="laptop")
        tm.that(query.category, eq="electronics")
        tm.that(query.min_price, none=False)
        if query.min_price is not None:
            tm.that(math.isclose(query.min_price, 500.0), eq=True)

    def test_domain_event_creation(self) -> None:
        event = m.DomainEvent(
            event_type="UserCreated",
            aggregate_id="USER-001",
            data=m.ComparableConfigMap(
                root={"user_id": "USER-001", "email": "user@example.com"},
            ),
        )
        tm.that(event.event_type, eq="UserCreated")
        tm.that(event.aggregate_id, eq="USER-001")
        tm.that(event.data["user_id"], eq="USER-001")
        tm.that(event.data["email"], eq="user@example.com")
        tm.that(event.unique_id, none=False)
        tm.that(event.created_at, none=False)

    def test_domain_event_equality(self) -> None:
        event1 = m.DomainEvent(
            event_type="OrderShipped",
            aggregate_id="ORD-001",
            data=m.ComparableConfigMap(root={"tracking_number": "TRACK-123"}),
        )
        event2 = m.DomainEvent(
            event_type="OrderShipped",
            aggregate_id="ORD-001",
            data=m.ComparableConfigMap(root={"tracking_number": "TRACK-123"}),
        )
        tm.that(event1.unique_id, ne=event2.unique_id)
        tm.that(event1.event_type, eq=event2.event_type)
        tm.that(event1.aggregate_id, eq=event2.aggregate_id)

    def test_domain_event_timestamp(self) -> None:
        event = m.DomainEvent(
            event_type="AccountUpdated",
            aggregate_id="ACC-001",
            data=m.ComparableConfigMap(root={"field": "balance"}),
        )
        tm.that(event.created_at, none=False)
        tm.that(event.created_at, is_=datetime)

    def test_domain_event_causality(self) -> None:
        event = m.DomainEvent(
            event_type="PaymentProcessed",
            aggregate_id="PAY-001",
            data=m.ComparableConfigMap(root={"amount": 99.99}),
        )
        tm.that(event.unique_id, none=False)
        tm.that(event.created_at, none=False)

    def test_metadata_creation(self) -> None:
        metadata = m.Metadata(attributes={"user_id": "123", "operation": "create"})
        tm.that(metadata.attributes["user_id"], eq="123")

    def test_metadata_with_various_types(self) -> None:
        metadata = m.Metadata(
            attributes={
                "string": "value",
                "number": 42,
                "float": math.pi,
                "bool": True,
            },
        )
        tm.that(metadata.attributes["string"], eq="value")
        tm.that(metadata.attributes["number"], eq=42)

    def test_model_validation_error_handling(self) -> None:
        class ValidatedEntity(m.Entity):
            email: str
            age: int

            @field_validator("age")
            @classmethod
            def validate_age(cls, value: int) -> int:
                if value < 0 or value > 150:
                    error_msg = "Invalid age"
                    raise ValueError(error_msg)
                return value

        entity = ValidatedEntity(email="user@example.com", age=30, domain_events=[])
        tm.that(entity.age, eq=30)
        with pytest.raises(ValidationError):
            ValidatedEntity(email="user@example.com", age=200, domain_events=[])

    def test_multiple_field_validation(self) -> None:
        class Profile(m.Entity):
            username: str
            email: str
            bio: str | None = None

            @field_validator("username")
            @classmethod
            def validate_username(cls, value: str) -> str:
                if len(value) < 3:
                    error_msg = "Username too short"
                    raise ValueError(error_msg)
                return value

            @field_validator("email")
            @classmethod
            def validate_email(cls, value: str) -> str:
                if "@" not in value:
                    error_msg = "Invalid email"
                    raise ValueError(error_msg)
                return value

        profile = Profile(
            username="alice",
            email="alice@example.com",
            bio="Developer",
            domain_events=[],
        )
        tm.that(profile.username, eq="alice")

    def test_entity_model_dump(self) -> None:
        class Task(m.Entity):
            title: str
            completed: bool

        task = Task(title="Complete tests", completed=False, domain_events=[])
        dumped = task.model_dump()
        tm.that(dumped["title"], eq="Complete tests")
        tm.that(not dumped["completed"], eq=True)
        tm.that(dumped, has="unique_id")

    def test_command_serialization(self) -> None:
        class SendEmailCommand(m.Command):
            recipient: str
            subject: str
            body: str

        cmd = SendEmailCommand(
            recipient="user@example.com",
            subject="Test",
            body="Message body",
            command_id="cmd-test-5",
        )
        dumped = cmd.model_dump()
        tm.that(dumped["recipient"], eq="user@example.com")
        tm.that(dumped["subject"], eq="Test")

    def test_aggregate_root_serialization(self) -> None:
        class ShoppingCart(m.AggregateRoot):
            items: Sequence[t.HeaderMapping]
            total: float

        cart = ShoppingCart(
            items=[
                {"product_id": "P1", "quantity": 2},
                {"product_id": "P2", "quantity": 1},
            ],
            total=99.99,
            domain_events=[],
        )
        dumped = cart.model_dump()
        tm.that(len(dumped["items"]), eq=2)
        tm.that(math.isclose(dumped["total"], 99.99), eq=True)

    def test_entity_model_validation(self) -> None:
        class Customer(m.Entity):
            name: str
            email: str

        customer = Customer(name="John", email="john@example.com", domain_events=[])
        dumped = customer.model_dump()
        customer_dict = {
            k: dumped[k] for k in type(customer).model_fields if k in dumped
        }
        validated = Customer.model_validate(customer_dict)
        tm.that(validated, none=False)
        tm.that(validated.name, eq="John")
        tm.that(validated.email, eq="john@example.com")

    def test_command_factory_pattern(self) -> None:
        class RegisterUserCommand(m.Command):
            email: str
            password: str

        cmd = RegisterUserCommand(
            email="user@example.com",
            password="secure123",
            command_id="cmd-test-6",
        )
        tm.that(cmd.email, eq="user@example.com")


__all__: list[str] = ["TestCoverageModels"]
