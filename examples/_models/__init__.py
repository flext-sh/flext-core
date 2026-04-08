# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".ex00": (
            "Ex00UserInput",
            "Ex00UserProfile",
        ),
        ".ex01": (
            "Ex01DemonstrationResult",
            "Ex01InvalidPersonPayload",
            "Ex01RunDemonstrationCommand",
            "Ex01User",
            "Ex01ValidPersonPayload",
        ),
        ".ex02": (
            "Ex02CacheService",
            "Ex02DatabaseService",
            "Ex02EmailService",
            "Ex02TestConfig",
        ),
        ".ex03": (
            "Ex03Email",
            "Ex03Money",
            "Ex03Order",
            "Ex03OrderItem",
            "Ex03User",
        ),
        ".ex04": (
            "Ex04AutoCommand",
            "Ex04CreateUser",
            "Ex04DeleteUser",
            "Ex04FailingDelete",
            "Ex04GetUser",
            "Ex04NoSubscriberEvent",
            "Ex04Ping",
            "Ex04UnknownQuery",
            "Ex04UserCreated",
        ),
        ".ex05": (
            "Ex05BadProcessor",
            "Ex05GoodProcessor",
            "Ex05HandlerBad",
            "Ex05HandlerLike",
            "Ex05StatusEnum",
            "Ex05UserModel",
        ),
        ".ex07": (
            "Ex07CreateUserCommand",
            "Ex07DemoPlugin",
            "Ex07GetUserQuery",
            "Ex07UserCreatedEvent",
        ),
        ".ex08": (
            "Ex08Order",
            "Ex08User",
        ),
        ".ex10": (
            "Ex10CommandBusStub",
            "Ex10ContextPayload",
            "Ex10DerivedMessage",
            "Ex10Entity",
            "Ex10Message",
            "Ex10ProcessorBad",
            "Ex10ProcessorGood",
            "Ex10ProtocolHandler",
            "Ex10ServiceStub",
        ),
        ".ex11": (
            "Ex11CommandBusStub",
            "Ex11EntityStub",
            "Ex11HandlerLike",
            "Ex11HandlerLikeService",
            "Ex11Payload",
            "Ex11ProcessorProtocolBad",
            "Ex11ProcessorProtocolGood",
        ),
        ".ex12": (
            "Ex12CommandA",
            "Ex12CommandB",
        ),
        ".ex14": (
            "Ex14CreateUserCommand",
            "Ex14GetUserQuery",
            "Ex14UserDTO",
        ),
        ".exconfig": ("ExConfigAppConfig",),
        ".shared": (
            "SharedHandle",
            "SharedPerson",
        ),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
