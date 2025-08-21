"""Script de teste para verificar se o warning de reportPrivateUsage foi suprimido."""

from flext_core import FlextCore


def test_private_access() -> None:
    """Testa o acesso a membro protegido que deveria gerar warning antes."""
    # Reset do singleton para garantir estado limpo
    FlextCore._instance = None  # Isso deveria não gerar warning agora

    # Criação de instância
    FlextCore.get_instance()


if __name__ == "__main__":
    test_private_access()
