"""Notification service for FLEXT Core."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class NotificationChannel(Enum):
    """Notification channels."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"


class NotificationPriority(Enum):
    """Notification priorities."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationService(ABC):
    """Abstract notification service."""

    @abstractmethod
    async def send_notification(self,
                               channel: NotificationChannel,
                               recipient: str,
                               subject: str,
                               message: str,
                               priority: NotificationPriority = NotificationPriority.MEDIUM) -> bool:
        """Send a notification."""

    @abstractmethod
    async def send_pipeline_notification(self,
                                       pipeline_id: str,
                                       event_type: str,
                                       message: str,
                                       priority: NotificationPriority = NotificationPriority.MEDIUM) -> bool:
        """Send pipeline-related notification."""

    @abstractmethod
    async def configure_notification_rules(self, rules: dict[str, Any]) -> None:
        """Configure notification rules."""


class DefaultNotificationService(NotificationService):
    """Default implementation of notification service."""

    def __init__(self) -> None:
        self._sent_notifications: list[dict[str, Any]] = []
        self._rules: dict[str, Any] = {}

    async def send_notification(self,
                               channel: NotificationChannel,
                               recipient: str,
                               subject: str,
                               message: str,
                               priority: NotificationPriority = NotificationPriority.MEDIUM) -> bool:
        """Send a notification."""
        notification = {
            "channel": channel.value,
            "recipient": recipient,
            "subject": subject,
            "message": message,
            "priority": priority.value,
            "sent_at": None,  # Would use actual timestamp
        }
        self._sent_notifications.append(notification)
        return True

    async def send_pipeline_notification(self,
                                       pipeline_id: str,
                                       event_type: str,
                                       message: str,
                                       priority: NotificationPriority = NotificationPriority.MEDIUM) -> bool:
        """Send pipeline-related notification."""
        return await self.send_notification(
            channel=NotificationChannel.EMAIL,  # Default channel
            recipient="REDACTED_LDAP_BIND_PASSWORD@example.com",  # Default recipient
            subject=f"Pipeline {event_type}: {pipeline_id}",
            message=message,
            priority=priority,
        )

    async def configure_notification_rules(self, rules: dict[str, Any]) -> None:
        """Configure notification rules."""
        self._rules.update(rules)
