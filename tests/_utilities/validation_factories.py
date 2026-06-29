"""Validation and failing service factory helpers for flext-core tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, override

from tests import c
from tests._utilities.services import TestsFlextUtilitiesServicesMixin
from tests._utilities.user_factories import TestsFlextUtilitiesUserFactoriesMixin


class TestsFlextUtilitiesValidationFactoriesMixin(
    TestsFlextUtilitiesServicesMixin, TestsFlextUtilitiesUserFactoriesMixin
):
    """Validation and failing service factory helpers."""

    class _FailingFactoryBase[T]:
        """Shared constructor contract for failing-service factories."""

        @classmethod
        def build(
            cls,
            *,
            error_message: str = c.Tests.DEFAULT_ERROR_MESSAGE,
        ) -> T:
            """Build a failing-service instance; subclasses provide the type."""
            raise NotImplementedError

        @classmethod
        def build_batch(cls, size: int) -> list[T]:
            """Build multiple failing-service instances."""
            return [cls.build() for _ in range(size)]

    class FailingServiceFactory(
        _FailingFactoryBase[TestsFlextUtilitiesServicesMixin.FailingService]
    ):
        """Factory for FailingService."""

        @classmethod
        @override
        def build(
            cls,
            *,
            error_message: str = c.Tests.DEFAULT_ERROR_MESSAGE,
        ) -> TestsFlextUtilitiesValidationFactoriesMixin.FailingService:
            """Build a FailingService instance."""
            return TestsFlextUtilitiesValidationFactoriesMixin.FailingService(
                error_message=error_message,
            )

    class GetUserServiceAutoFactory(
        TestsFlextUtilitiesUserFactoriesMixin._GetUserFactoryBase[
            TestsFlextUtilitiesServicesMixin.GetUserServiceAuto
        ]
    ):
        """Factory for GetUserServiceAuto."""

        @classmethod
        @override
        def build(
            cls, *, user_id: str | None = None
        ) -> TestsFlextUtilitiesValidationFactoriesMixin.GetUserServiceAuto:
            """Build a GetUserServiceAuto instance."""
            return TestsFlextUtilitiesValidationFactoriesMixin.GetUserServiceAuto(
                user_id=cls._resolve_user_id(user_id),
            )

    class _ValidatingFactoryBase[T]:
        """Shared word-rotating state for validating-service factories.

        Subclasses override ``_make_instance`` with the typed return;
        the base owns ``_words``, ``_word_index``, ``_next_word``,
        ``build``, ``build_batch``, and ``reset``.
        """

        _words: ClassVar[Sequence[str]] = (
            "alpha",
            "bravo",
            "charlie",
            "delta",
            "echo",
        )
        _word_index: ClassVar[int] = 0

        @classmethod
        def _next_word(cls) -> str:
            """Get next word from rotation."""
            word = cls._words[cls._word_index % len(cls._words)]
            cls._word_index += 1
            return word

        @classmethod
        def _make_instance(cls, value_input: str, min_length: int) -> T:
            """Construct one instance; subclasses override with typed return."""
            raise NotImplementedError

        @classmethod
        def build(
            cls,
            *,
            value_input: str | None = None,
            min_length: int = c.Tests.MIN_LENGTH_DEFAULT,
        ) -> T:
            """Build one validating-service instance."""
            actual_value = value_input if value_input is not None else cls._next_word()
            return cls._make_instance(actual_value, min_length)

        @classmethod
        def build_batch(cls, size: int) -> list[T]:
            """Build multiple validating-service instances."""
            return [cls.build() for _ in range(size)]

        @classmethod
        def reset(cls) -> None:
            """Reset per-subclass factory counter."""
            cls._word_index = 0

    class ValidatingServiceAutoFactory(
        _ValidatingFactoryBase[TestsFlextUtilitiesServicesMixin.ValidatingServiceAuto]
    ):
        """Factory for ValidatingServiceAuto."""

        @classmethod
        @override
        def _make_instance(
            cls, value_input: str, min_length: int
        ) -> TestsFlextUtilitiesValidationFactoriesMixin.ValidatingServiceAuto:
            """Construct a ValidatingServiceAuto instance."""
            return TestsFlextUtilitiesValidationFactoriesMixin.ValidatingServiceAuto(
                value_input=value_input,
                min_length=min_length,
            )

    class ValidatingServiceFactory(
        _ValidatingFactoryBase[TestsFlextUtilitiesServicesMixin.ValidatingService]
    ):
        """Factory for ``ValidatingService``."""

        @classmethod
        @override
        def _make_instance(
            cls, value_input: str, min_length: int
        ) -> TestsFlextUtilitiesValidationFactoriesMixin.ValidatingService:
            """Construct a ValidatingService instance."""
            return TestsFlextUtilitiesValidationFactoriesMixin.ValidatingService(
                value_input=value_input,
                min_length=min_length,
            )

    class FailingServiceAutoFactory(
        _FailingFactoryBase[TestsFlextUtilitiesServicesMixin.FailingServiceAuto]
    ):
        """Factory for FailingServiceAuto."""

        @classmethod
        @override
        def build(
            cls,
            *,
            error_message: str = c.Tests.DEFAULT_ERROR_MESSAGE,
        ) -> TestsFlextUtilitiesValidationFactoriesMixin.FailingServiceAuto:
            """Build a FailingServiceAuto instance."""
            return TestsFlextUtilitiesValidationFactoriesMixin.FailingServiceAuto(
                error_message=error_message,
            )


__all__: list[str] = ["TestsFlextUtilitiesValidationFactoriesMixin"]
