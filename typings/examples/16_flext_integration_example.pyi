from .shared_domain import (
    EmailAddress as EmailAddress,
    Money as Money,
    Order as Order,
    SharedDomainFactory as SharedDomainFactory,
    User as User,
)

MIN_EMAIL_LENGTH: int

class _OrderRepositoryProtocol:
    def save(self, order: Order) -> object: ...

def main() -> None: ...
