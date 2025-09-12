"""Simple test coverage for decorators.py module."""

from flext_core import FlextDecorators


class TestDecoratorsSimple:
    """Test FlextDecorators basic functionality."""

    def test_decorators_class_exists(self) -> None:
        """Test FlextDecorators class exists."""
        assert FlextDecorators is not None

    def test_retry_decorator_works(self) -> None:
        """Test retry decorator works."""
        # Test it returns a decorator
        decorator = FlextDecorators.retry(max_attempts=3)
        assert callable(decorator)

        # Test it doesn't break functions
        @FlextDecorators.retry(max_attempts=3)
        def test_func(x: int) -> int:
            return x * 2

        assert test_func(5) == 10

    def test_timeout_decorator_works(self) -> None:
        """Test timeout decorator works."""
        # Test it returns a decorator
        decorator = FlextDecorators.timeout(seconds=10)
        assert callable(decorator)

        # Test it doesn't break functions
        @FlextDecorators.timeout(seconds=10)
        def test_func(x: int) -> int:
            return x + 5

        assert test_func(3) == 8

    def test_cache_decorator_works(self) -> None:
        """Test cache decorator works."""
        # Test it returns a decorator
        decorator = FlextDecorators.cache(ttl=300)
        assert callable(decorator)

        # Test it doesn't break functions
        @FlextDecorators.cache(ttl=300)
        def test_func(x: int) -> int:
            return x * 3

        assert test_func(4) == 12
