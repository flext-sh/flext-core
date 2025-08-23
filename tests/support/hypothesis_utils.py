"""Advanced property-based testing utilities using Hypothesis.

Provides comprehensive property-based testing strategies, custom generators,
and sophisticated test data generation patterns for FlextCore testing.
"""

from __future__ import annotations

import string
from datetime import UTC, datetime
from typing import Any, TypeVar
from uuid import uuid4

from hypothesis import strategies as st

T = TypeVar("T")


class FlextStrategies:
    """Custom Hypothesis strategies for Flext-specific data types."""

    @staticmethod
    def flext_ids() -> st.SearchStrategy[str]:
        """Generate Flext-style IDs."""
        return st.builds(
            lambda: f"flext_{uuid4().hex[:8]}",
        )

    @staticmethod
    def correlation_ids() -> st.SearchStrategy[str]:
        """Generate correlation IDs."""
        return st.builds(
            lambda prefix, suffix: f"{prefix}_{suffix}",
            prefix=st.sampled_from(["corr", "trace", "req", "op"]),
            suffix=st.text(
                alphabet=string.ascii_lowercase + string.digits,
                min_size=8,
                max_size=16,
            ),
        )

    @staticmethod
    def emails() -> st.SearchStrategy[str]:
        """Generate realistic email addresses."""
        return st.builds(
            lambda local, domain, tld: f"{local}@{domain}.{tld}",
            local=st.text(
                alphabet=string.ascii_lowercase + string.digits + "._-",
                min_size=1,
                max_size=20,
            ).filter(lambda x: x[0] not in "._-" and x[-1] not in "._-"),
            domain=st.text(
                alphabet=string.ascii_lowercase + string.digits + "-",
                min_size=1,
                max_size=20,
            ).filter(lambda x: x[0] != "-" and x[-1] != "-"),
            tld=st.sampled_from(["com", "org", "net", "edu", "gov", "co.uk", "io"]),
        )

    @staticmethod
    def phone_numbers() -> st.SearchStrategy[str]:
        """Generate phone numbers in various formats."""
        return st.one_of(
            [
                # +1-555-123-4567
                st.builds(
                    lambda area, exchange, number: f"+1-{area}-{exchange}-{number}",
                    area=st.integers(min_value=200, max_value=999).map(str),
                    exchange=st.integers(min_value=200, max_value=999).map(str),
                    number=st.integers(min_value=1000, max_value=9999).map(str),
                ),
                # (555) 123-4567
                st.builds(
                    lambda area, exchange, number: f"({area}) {exchange}-{number}",
                    area=st.integers(min_value=200, max_value=999).map(str),
                    exchange=st.integers(min_value=200, max_value=999).map(str),
                    number=st.integers(min_value=1000, max_value=9999).map(str),
                ),
                # 555.123.4567
                st.builds(
                    lambda area, exchange, number: f"{area}.{exchange}.{number}",
                    area=st.integers(min_value=200, max_value=999).map(str),
                    exchange=st.integers(min_value=200, max_value=999).map(str),
                    number=st.integers(min_value=1000, max_value=9999).map(str),
                ),
            ]
        )

    @staticmethod
    def urls() -> st.SearchStrategy[str]:
        """Generate various URL formats."""
        return st.builds(
            lambda scheme, domain, tld, path: f"{scheme}://{domain}.{tld}{path}",
            scheme=st.sampled_from(["http", "https"]),
            domain=st.text(
                alphabet=string.ascii_lowercase + string.digits,
                min_size=3,
                max_size=20,
            ),
            tld=st.sampled_from(["com", "org", "net", "io", "co.uk"]),
            path=st.one_of(
                [
                    st.just(""),
                    st.builds(
                        lambda p: f"/{p}",
                        p=st.text(
                            alphabet=string.ascii_lowercase + string.digits + "-_/",
                            min_size=1,
                            max_size=50,
                        ),
                    ),
                ]
            ),
        )

    @staticmethod
    def timestamps() -> st.SearchStrategy[datetime]:
        """Generate timestamps within reasonable ranges."""
        return st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2030, 12, 31),
            timezones=st.just(UTC),
        )

    @staticmethod
    def iso_timestamps() -> st.SearchStrategy[str]:
        """Generate ISO formatted timestamp strings."""
        return FlextStrategies.timestamps().map(lambda dt: dt.isoformat())


class BusinessDomainStrategies:
    """Strategies for business domain objects."""

    @staticmethod
    def user_names() -> st.SearchStrategy[str]:
        """Generate realistic user names."""
        first_names = [
            "John",
            "Jane",
            "Michael",
            "Sarah",
            "David",
            "Emily",
            "Robert",
            "Jessica",
            "William",
            "Ashley",
            "James",
            "Amanda",
        ]
        last_names = [
            "Smith",
            "Johnson",
            "Brown",
            "Davis",
            "Miller",
            "Wilson",
            "Moore",
            "Taylor",
            "Anderson",
            "Thomas",
            "Jackson",
            "White",
        ]

        return st.builds(
            lambda first, last: f"{first} {last}",
            first=st.sampled_from(first_names),
            last=st.sampled_from(last_names),
        )

    @staticmethod
    def company_names() -> st.SearchStrategy[str]:
        """Generate company names."""
        words = [
            "Tech",
            "Solutions",
            "Systems",
            "Digital",
            "Global",
            "Advanced",
            "Innovative",
            "Dynamic",
            "Strategic",
            "Professional",
            "Enterprise",
        ]
        suffixes = ["Inc", "LLC", "Corp", "Ltd", "Co"]

        return st.builds(
            lambda word1, word2, suffix: f"{word1} {word2} {suffix}",
            word1=st.sampled_from(words),
            word2=st.sampled_from(words),
            suffix=st.sampled_from(suffixes),
        ).filter(
            lambda x: len(x.split()[0]) != len(x.split()[1])
        )  # Avoid duplicate words

    @staticmethod
    def addresses() -> st.SearchStrategy[dict[str, str]]:
        """Generate address objects."""
        street_names = [
            "Main St",
            "Oak Ave",
            "Pine Rd",
            "First St",
            "Second Ave",
            "Park Blvd",
            "Washington St",
            "Lincoln Ave",
            "Jefferson Rd",
        ]
        cities = [
            "Springfield",
            "Madison",
            "Georgetown",
            "Franklin",
            "Clinton",
            "Riverside",
            "Salem",
            "Auburn",
            "Bristol",
            "Camden",
        ]
        states = [
            "CA",
            "TX",
            "FL",
            "NY",
            "PA",
            "IL",
            "OH",
            "GA",
            "NC",
            "MI",
        ]

        return st.builds(
            dict,
            street=st.builds(
                lambda num, name: f"{num} {name}",
                num=st.integers(min_value=1, max_value=9999),
                name=st.sampled_from(street_names),
            ),
            city=st.sampled_from(cities),
            state=st.sampled_from(states),
            zip_code=st.builds(
                lambda z: f"{z:05d}",
                z=st.integers(min_value=10000, max_value=99999),
            ),
        )


class EdgeCaseStrategies:
    """Strategies for edge cases and boundary conditions."""

    @staticmethod
    def empty_or_whitespace_strings() -> st.SearchStrategy[str]:
        """Generate empty or whitespace-only strings."""
        return st.one_of(
            [
                st.just(""),
                st.text(alphabet=" \t\n\r", min_size=1, max_size=10),
                st.builds(lambda n: " " * n, n=st.integers(min_value=1, max_value=100)),
            ]
        )

    @staticmethod
    def boundary_integers() -> st.SearchStrategy[int]:
        """Generate integers at common boundary values."""
        return st.one_of(
            [
                st.just(0),
                st.just(1),
                st.just(-1),
                st.integers(
                    min_value=-(2**31), max_value=-(2**31) + 10
                ),  # Near min int32
                st.integers(
                    min_value=2**31 - 10, max_value=2**31 - 1
                ),  # Near max int32
                st.integers(
                    min_value=-(2**63), max_value=-(2**63) + 10
                ),  # Near min int64
                st.integers(
                    min_value=2**63 - 10, max_value=2**63 - 1
                ),  # Near max int64
            ]
        )

    @staticmethod
    def boundary_floats() -> st.SearchStrategy[float]:
        """Generate floats at boundary values."""
        return st.one_of(
            [
                st.just(0.0),
                st.just(-0.0),
                st.just(1.0),
                st.just(-1.0),
                st.floats(min_value=0.0, max_value=1e-10),  # Very small positive
                st.floats(min_value=-1e-10, max_value=0.0),  # Very small negative
                st.floats(min_value=1e10, max_value=1e15),  # Very large positive
                st.floats(min_value=-1e15, max_value=-1e10),  # Very large negative
            ]
        )

    @staticmethod
    def unicode_edge_cases() -> st.SearchStrategy[str]:
        """Generate Unicode edge cases."""
        return st.one_of(
            [
                st.text(alphabet="🚀🎯✅❌🔧📊", min_size=1, max_size=10),  # Emojis
                st.text(alphabet="áéíóúñü", min_size=1, max_size=20),  # Accented chars
                st.text(alphabet="αβγδεζηθ", min_size=1, max_size=15),  # Greek letters
                st.text(
                    alphabet="中文测试", min_size=1, max_size=10
                ),  # Chinese characters
                st.builds(lambda: "\u200b" * 5),  # Zero-width spaces
                st.builds(lambda: "test\x00null"),  # Null characters
            ]
        )

    @staticmethod
    def malformed_data() -> st.SearchStrategy[dict[str, Any]]:
        """Generate malformed data structures."""
        return st.one_of(
            [
                st.just({}),  # Empty dict
                st.builds(dict, key1=st.none(), key2=st.none()),  # None values
                st.builds(
                    dict,
                    **{
                        f"key_{i}": st.one_of([st.none(), st.just(""), st.integers()])
                        for i in range(5)
                    },
                ),
                st.recursive(
                    st.none() | st.booleans() | st.integers() | st.text(),
                    lambda children: st.lists(children)
                    | st.dictionaries(st.text(), children),
                    max_leaves=20,
                ),
            ]
        )


class PerformanceStrategies:
    """Strategies for performance testing."""

    @staticmethod
    def large_strings() -> st.SearchStrategy[str]:
        """Generate large strings for performance testing."""
        return st.builds(
            lambda size, char: char * size,
            size=st.integers(min_value=1000, max_value=100000),
            char=st.sampled_from(string.ascii_letters),
        )

    @staticmethod
    def large_lists() -> st.SearchStrategy[list[int]]:
        """Generate large lists for performance testing."""
        return st.lists(
            st.integers(min_value=0, max_value=1000000),
            min_size=1000,
            max_size=10000,
        )

    @staticmethod
    def nested_structures() -> st.SearchStrategy[dict[str, Any]]:
        """Generate deeply nested structures."""
        return st.recursive(
            st.builds(dict, key=st.text(), value=st.integers()),
            lambda children: st.dictionaries(
                st.text(min_size=1, max_size=10),
                st.one_of([children, st.lists(children, max_size=5)]),
            ),
            max_leaves=50,
        )


class PropertyTestHelpers:
    """Helper functions for property-based testing."""

    @staticmethod
    def assume_valid_email(email: str) -> bool:
        """Check if an email meets basic validation assumptions."""
        return (
            "@" in email
            and len(email) > 3
            and not email.startswith("@")
            and not email.endswith("@")
            and ".." not in email
        )

    @staticmethod
    def assume_non_empty_string(text: str) -> bool:
        """Check if string is non-empty after stripping."""
        return len(text.strip()) > 0

    @staticmethod
    def assume_positive_number(num: float) -> bool:
        """Check if number is positive."""
        return num > 0 and num == num  # Not NaN

    @staticmethod
    def assume_valid_url(url: str) -> bool:
        """Check if URL meets basic validation assumptions."""
        return (
            "://" in url
            and len(url) > 7  # Minimum: "http://"
            and not url.endswith("://")
            and url.startswith(("http://", "https://"))
        )

    @staticmethod
    def generate_test_scenarios(
        strategy: st.SearchStrategy[T],
        scenario_name: str = "test_scenario",
        min_examples: int = 10,
    ) -> st.SearchStrategy[dict[str, Any]]:
        """Generate test scenarios with metadata."""
        return st.builds(
            lambda data, id_val: {
                "scenario": scenario_name,
                "data": data,
                "id": id_val,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            data=strategy,
            id_val=FlextStrategies.flext_ids(),
        )


# Pre-configured composite strategies for common use cases
class CompositeStrategies:
    """Pre-configured composite strategies for common testing scenarios."""

    @staticmethod
    def user_profiles() -> st.SearchStrategy[dict[str, Any]]:
        """Complete user profile data."""
        return st.builds(
            dict,
            id=FlextStrategies.flext_ids(),
            name=BusinessDomainStrategies.user_names(),
            email=FlextStrategies.emails(),
            phone=FlextStrategies.phone_numbers(),
            address=BusinessDomainStrategies.addresses(),
            created_at=FlextStrategies.iso_timestamps(),
            active=st.booleans(),
            metadata=st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.one_of([st.text(), st.integers(), st.booleans()]),
                max_size=5,
            ),
        )

    @staticmethod
    def api_requests() -> st.SearchStrategy[dict[str, Any]]:
        """API request-like data structures."""
        return st.builds(
            dict,
            method=st.sampled_from(["GET", "POST", "PUT", "DELETE", "PATCH"]),
            url=FlextStrategies.urls(),
            headers=st.dictionaries(
                st.sampled_from(
                    [
                        "Content-Type",
                        "Authorization",
                        "Accept",
                        "User-Agent",
                        "X-Correlation-ID",
                    ]
                ),
                st.text(min_size=1, max_size=100),
                max_size=5,
            ),
            body=st.one_of(
                [
                    st.none(),
                    st.dictionaries(st.text(), st.text(), max_size=10),
                    st.text(max_size=1000),
                ]
            ),
            correlation_id=FlextStrategies.correlation_ids(),
        )

    @staticmethod
    def configuration_data() -> st.SearchStrategy[dict[str, Any]]:
        """Configuration-like data structures."""
        return st.builds(
            dict,
            database_url=st.builds(
                lambda host, port, db: f"postgresql://{host}:{port}/{db}",
                host=st.text(alphabet=string.ascii_lowercase, min_size=5, max_size=20),
                port=st.integers(min_value=1000, max_value=65535),
                db=st.text(
                    alphabet=string.ascii_lowercase + "_", min_size=3, max_size=20
                ),
            ),
            debug=st.booleans(),
            timeout_seconds=st.integers(min_value=1, max_value=300),
            retry_attempts=st.integers(min_value=0, max_value=10),
            features=st.lists(
                st.text(alphabet=string.ascii_lowercase + "_", min_size=3, max_size=30),
                max_size=10,
            ),
            environment=st.sampled_from(["development", "staging", "production"]),
        )
