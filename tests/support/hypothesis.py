"""Advanced property-based testing utilities using Hypothesis.

Provides comprehensive property-based testing strategies, custom generators,
and sophisticated test data generation patterns for FlextCore testing.
"""

from __future__ import annotations

import math
import string
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TypeVar
from uuid import uuid4

from hypothesis import strategies as st

T = TypeVar("T")

# Constants
MIN_EMAIL_LENGTH = 3
MIN_URL_LENGTH = 7


class FlextStrategies:
    """Custom Hypothesis strategies for Flext-specific data types."""

    @staticmethod
    def _generate_flext_id() -> str:
        """Generate a Flext-style ID."""
        return f"flext_{uuid4().hex[:8]}"

    @staticmethod
    def _build_correlation_id(prefix: str, suffix: str) -> str:
        """Build correlation ID from prefix and suffix."""
        return f"{prefix}_{suffix}"

    @staticmethod
    def _build_email(local: str, domain: str, tld: str) -> str:
        """Build email address from components."""
        return f"{local}@{domain}.{tld}"

    @staticmethod
    def _is_valid_email_part(x: str) -> bool:
        """Check if email part is valid (no leading/trailing dots/dashes)."""
        return x[0] not in "._-" and x[-1] not in "._-"

    @staticmethod
    def _is_valid_domain(x: str) -> bool:
        """Check if domain is valid (no leading/trailing dashes)."""
        return x[0] != "-" and x[-1] != "-"

    @staticmethod
    def flext_ids() -> st.SearchStrategy[str]:
        """Generate Flext-style IDs."""
        return st.builds(FlextStrategies._generate_flext_id)

    @staticmethod
    def correlation_ids() -> st.SearchStrategy[str]:
        """Generate correlation IDs."""
        return st.builds(
            FlextStrategies._build_correlation_id,
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
            FlextStrategies._build_email,
            local=st.text(
                alphabet=string.ascii_lowercase + string.digits + "._-",
                min_size=1,
                max_size=20,
            ).filter(FlextStrategies._is_valid_email_part),
            domain=st.text(
                alphabet=string.ascii_lowercase + string.digits + "-",
                min_size=1,
                max_size=20,
            ).filter(FlextStrategies._is_valid_domain),
            tld=st.sampled_from(["com", "org", "net", "edu", "gov", "co.uk", "io"]),
        )

    @staticmethod
    def _format_phone_international(area: str, exchange: str, number: str) -> str:
        """Format phone number in international format (+1-555-123-4567)."""
        return f"+1-{area}-{exchange}-{number}"

    @staticmethod
    def _format_phone_parentheses(area: str, exchange: str, number: str) -> str:
        """Format phone number with parentheses ((555) 123-4567)."""
        return f"({area}) {exchange}-{number}"

    @staticmethod
    def _format_phone_dotted(area: str, exchange: str, number: str) -> str:
        """Format phone number with dots (555.123.4567)."""
        return f"{area}.{exchange}.{number}"

    @staticmethod
    def phone_numbers() -> st.SearchStrategy[str]:
        """Generate phone numbers in various formats."""
        return st.one_of(
            [
                # +1-555-123-4567
                st.builds(
                    FlextStrategies._format_phone_international,
                    area=st.integers(min_value=200, max_value=999).map(str),
                    exchange=st.integers(min_value=200, max_value=999).map(str),
                    number=st.integers(min_value=1000, max_value=9999).map(str),
                ),
                # (555) 123-4567
                st.builds(
                    FlextStrategies._format_phone_parentheses,
                    area=st.integers(min_value=200, max_value=999).map(str),
                    exchange=st.integers(min_value=200, max_value=999).map(str),
                    number=st.integers(min_value=1000, max_value=9999).map(str),
                ),
                # 555.123.4567
                st.builds(
                    FlextStrategies._format_phone_dotted,
                    area=st.integers(min_value=200, max_value=999).map(str),
                    exchange=st.integers(min_value=200, max_value=999).map(str),
                    number=st.integers(min_value=1000, max_value=9999).map(str),
                ),
            ]
        )

    @staticmethod
    def _build_url(scheme: str, domain: str, tld: str, path: str) -> str:
        """Build URL from components."""
        return f"{scheme}://{domain}.{tld}{path}"

    @staticmethod
    def _build_path(p: str) -> str:
        """Build URL path with leading slash."""
        return f"/{p}"

    @staticmethod
    def urls() -> st.SearchStrategy[str]:
        """Generate various URL formats."""
        return st.builds(
            FlextStrategies._build_url,
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
                        FlextStrategies._build_path,
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
            min_value=datetime(2020, 1, 1),  # noqa: DTZ001 - Required by Hypothesis when using timezones
            max_value=datetime(2030, 12, 31),  # noqa: DTZ001 - Required by Hypothesis when using timezones
            timezones=st.just(UTC),
        )

    @staticmethod
    def _to_isoformat(dt: datetime) -> str:
        """Convert datetime to ISO format string."""
        return dt.isoformat()

    @staticmethod
    def iso_timestamps() -> st.SearchStrategy[str]:
        """Generate ISO formatted timestamp strings."""
        return FlextStrategies.timestamps().map(FlextStrategies._to_isoformat)


class BusinessDomainStrategies:
    """Strategies for business domain objects."""

    @staticmethod
    def _build_full_name(first: str, last: str) -> str:
        """Build full name from first and last name."""
        return f"{first} {last}"

    @staticmethod
    def _build_company_name(word1: str, word2: str, suffix: str) -> str:
        """Build company name from components."""
        return f"{word1} {word2} {suffix}"

    @staticmethod
    def _has_different_words(x: str) -> bool:
        """Check if first two words are different (avoid duplicate words)."""
        words = x.split()
        return len(words) >= 2 and len(words[0]) != len(words[1])

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
            BusinessDomainStrategies._build_full_name,
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
            BusinessDomainStrategies._build_company_name,
            word1=st.sampled_from(words),
            word2=st.sampled_from(words),
            suffix=st.sampled_from(suffixes),
        ).filter(BusinessDomainStrategies._has_different_words)  # Avoid duplicate words

    @staticmethod
    def _build_street_address(num: int, name: str) -> str:
        """Build street address from number and name."""
        return f"{num} {name}"

    @staticmethod
    def _format_zip_code(z: int) -> str:
        """Format zip code with leading zeros."""
        return f"{z:05d}"

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

        return st.builds(  # pyright: ignore[reportUnknownVariableType]
            dict,
            street=st.builds(
                BusinessDomainStrategies._build_street_address,
                num=st.integers(min_value=1, max_value=9999),
                name=st.sampled_from(street_names),
            ),
            city=st.sampled_from(cities),
            state=st.sampled_from(states),
            zip_code=st.builds(
                BusinessDomainStrategies._format_zip_code,
                z=st.integers(min_value=10000, max_value=99999),
            ),
        )


class EdgeCaseStrategies:
    """Strategies for edge cases and boundary conditions."""

    @staticmethod
    def _repeat_spaces(n: int) -> str:
        """Repeat spaces n times."""
        return " " * n

    @staticmethod
    def _zero_width_spaces() -> str:
        """Generate zero-width space string."""
        return "\u200b" * 5

    @staticmethod
    def _null_character_string() -> str:
        """Generate string with null character."""
        return "test\x00null"

    @staticmethod
    def empty_or_whitespace_strings() -> st.SearchStrategy[str]:
        """Generate empty or whitespace-only strings."""
        return st.one_of(
            [
                st.just(""),
                st.text(alphabet=" \t\n\r", min_size=1, max_size=10),
                st.builds(
                    EdgeCaseStrategies._repeat_spaces,
                    n=st.integers(min_value=1, max_value=100),
                ),
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
                st.builds(EdgeCaseStrategies._zero_width_spaces),  # Zero-width spaces
                st.builds(EdgeCaseStrategies._null_character_string),  # Null characters
            ]
        )

    @staticmethod
    def malformed_data() -> st.SearchStrategy[dict[str, object]]:
        """Generate malformed data structures."""
        return st.one_of(
            [
                st.just({}),  # Empty dict
                st.dictionaries(
                    st.just("key1"), st.none(), min_size=2, max_size=2
                ),  # None values
                st.dictionaries(
                    st.sampled_from([f"key_{i}" for i in range(5)]),
                    st.one_of([st.none(), st.just(""), st.integers()]),
                    max_size=5,
                ),
                st.dictionaries(
                    st.text(),
                    st.one_of([st.none(), st.booleans(), st.integers(), st.text()]),
                    max_size=5,
                ),
            ]
        )


class PerformanceStrategies:
    """Strategies for performance testing."""

    @staticmethod
    def _repeat_character(size: int, char: str) -> str:
        """Repeat character to create large string."""
        return char * size

    @staticmethod
    def _build_nested_structure(
        children: st.SearchStrategy[Mapping[str, object]],
    ) -> st.SearchStrategy[Mapping[str, object]]:
        """Build nested dictionary structure recursively."""
        return st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.one_of([children, st.lists(children, max_size=5)]),
        )

    @staticmethod
    def large_strings() -> st.SearchStrategy[str]:
        """Generate large strings for performance testing."""
        return st.builds(
            PerformanceStrategies._repeat_character,
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
    def nested_structures() -> st.SearchStrategy[Mapping[str, object]]:
        """Generate deeply nested structures."""
        return st.recursive(
            st.dictionaries(
                st.text(),
                st.one_of([st.integers(), st.text(), st.booleans()]),
                max_size=1,
            ),
            PerformanceStrategies._build_nested_structure,
            max_leaves=50,
        )


class PropertyTestHelpers:
    """Helper functions for property-based testing."""

    @staticmethod
    def assume_valid_email(email: str) -> bool:
        """Check if an email meets basic validation assumptions."""
        return (
            "@" in email
            and len(email) > MIN_EMAIL_LENGTH
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
        return num > 0 and not math.isnan(num)

    @staticmethod
    def assume_valid_url(url: str) -> bool:
        """Check if URL meets basic validation assumptions."""
        return (
            "://" in url
            and len(url) > MIN_URL_LENGTH  # Minimum: "http://"
            and not url.endswith("://")
            and url.startswith(("http://", "https://"))
        )

    @staticmethod
    def _build_test_scenario(
        data: object, id_val: str, scenario_name: str
    ) -> dict[str, object]:
        """Build test scenario dictionary with metadata."""
        return {
            "scenario": scenario_name,
            "data": data,
            "id": id_val,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    @staticmethod
    def generate_test_scenarios(
        strategy: st.SearchStrategy[T],
        scenario_name: str = "test_scenario",
        _min_examples: int = 10,
    ) -> st.SearchStrategy[dict[str, object]]:
        """Generate test scenarios with metadata."""

        def _scenario_builder(data: T, id_val: str) -> dict[str, object]:
            return PropertyTestHelpers._build_test_scenario(data, id_val, scenario_name)

        return st.builds(
            _scenario_builder,
            data=strategy,
            id_val=FlextStrategies.flext_ids(),
        )


# Pre-configured composite strategies for common use cases
class CompositeStrategies:
    """Pre-configured composite strategies for common testing scenarios."""

    @staticmethod
    def user_profiles() -> st.SearchStrategy[dict[str, object]]:
        """Complete user profile data."""

        @st.composite
        def _user_profile(draw: st.DrawFn) -> dict[str, object]:
            return {
                "id": draw(FlextStrategies.flext_ids()),
                "name": draw(BusinessDomainStrategies.user_names()),
                "email": draw(FlextStrategies.emails()),
                "phone": draw(FlextStrategies.phone_numbers()),
                "address": draw(BusinessDomainStrategies.addresses()),
                "created_at": draw(FlextStrategies.iso_timestamps()),
                "active": draw(st.booleans()),
                "metadata": draw(
                    st.dictionaries(
                        st.text(min_size=1, max_size=20),
                        st.one_of([st.text(), st.integers(), st.booleans()]),
                        max_size=5,
                    )
                ),
            }

        return _user_profile()

    @staticmethod
    def api_requests() -> st.SearchStrategy[dict[str, object]]:
        """API request-like data structures."""

        @st.composite
        def _api_request(draw: st.DrawFn) -> dict[str, object]:
            return {
                "method": draw(
                    st.sampled_from(["GET", "POST", "PUT", "DELETE", "PATCH"])
                ),
                "url": draw(FlextStrategies.urls()),
                "headers": draw(
                    st.dictionaries(
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
                    )
                ),
                "body": draw(
                    st.one_of(
                        [
                            st.none(),
                            st.dictionaries(st.text(), st.text(), max_size=10),
                            st.text(max_size=1000),
                        ]
                    )
                ),
                "correlation_id": draw(FlextStrategies.correlation_ids()),
            }

        return _api_request()

    @staticmethod
    def _build_database_url(host: str, port: int, db: str) -> str:
        """Build PostgreSQL database URL."""
        return f"postgresql://{host}:{port}/{db}"

    @staticmethod
    def configuration_data() -> st.SearchStrategy[dict[str, object]]:
        """Configuration-like data structures."""

        @st.composite
        def _config_data(draw: st.DrawFn) -> dict[str, object]:
            return {
                "database_url": draw(
                    st.builds(
                        CompositeStrategies._build_database_url,
                        host=st.text(
                            alphabet=string.ascii_lowercase, min_size=5, max_size=20
                        ),
                        port=st.integers(min_value=1000, max_value=65535),
                        db=st.text(
                            alphabet=string.ascii_lowercase + "_",
                            min_size=3,
                            max_size=20,
                        ),
                    )
                ),
                "debug": draw(st.booleans()),
                "timeout_seconds": draw(st.integers(min_value=1, max_value=300)),
                "retry_attempts": draw(st.integers(min_value=0, max_value=10)),
                "features": draw(
                    st.lists(
                        st.text(
                            alphabet=string.ascii_lowercase + "_",
                            min_size=3,
                            max_size=30,
                        ),
                        max_size=10,
                    )
                ),
                "environment": draw(
                    st.sampled_from(["development", "staging", "production"])
                ),
            }

        return _config_data()
