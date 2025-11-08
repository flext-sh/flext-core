"""Testar get_args() para extrair tipo genérico.

OBJETIVO: FlextResult[int].ok("string") deve REJEITAR automaticamente.
"""

from typing import Generic, TypeVar, get_args, get_origin

from beartype import beartype
from beartype.door import is_bearable

T = TypeVar("T")


print("=" * 70)
print("TESTANDO get_args() para capturar [T]")
print("=" * 70)


@beartype
class FlextResultAutoValidate(Generic[T]):
    """FlextResult que extrai e valida tipo automaticamente."""

    def __init__(self, data: T):
        self._data = data

    @classmethod
    def ok(cls, data: T) -> "FlextResultAutoValidate[T]":
        """ok() que valida tipo automaticamente via get_args(cls)."""
        # DEBUG: Ver o que cls contém
        print("\nDEBUG ok():")
        print(f"  cls = {cls}")
        print(f"  type(cls) = {type(cls)}")
        print(f"  get_origin(cls) = {get_origin(cls)}")
        print(f"  get_args(cls) = {get_args(cls)}")

        # Tentar extrair tipo genérico
        args = get_args(cls)
        if args:
            expected_type = args[0]
            print(f"  ✅ Tipo extraído: {expected_type}")

            # Validar com is_bearable
            if not is_bearable(data, expected_type):
                raise TypeError(
                    f"FlextResult[{expected_type.__name__}].ok() received "
                    f"{type(data).__name__} instead of {expected_type.__name__}"
                )
            print(f"  ✅ Validação OK: data é {expected_type.__name__}")
        else:
            print("  ⚠️ Sem tipo genérico - pulando validação")

        return cls(data)


# =============================================================================
# TESTE 1: FlextResult[int].ok(42) - DEVE PASSAR
# =============================================================================
print("\n" + "=" * 70)
print("TESTE 1: FlextResult[int].ok(42) - tipo correto")
print("=" * 70)

try:
    result1 = FlextResultAutoValidate[int].ok(42)
    print(f"\n✅ PASSOU: {result1._data}")
except Exception as e:
    print(f"\n❌ FALHOU: {type(e).__name__}: {e}")

# =============================================================================
# TESTE 2: FlextResult[int].ok("string") - DEVE REJEITAR
# =============================================================================
print("\n" + "=" * 70)
print("TESTE 2: FlextResult[int].ok('string') - tipo ERRADO")
print("=" * 70)

try:
    result2 = FlextResultAutoValidate[int].ok("string")
    print(f"\n❌ PASSOU (deveria ter rejeitado!): {result2._data}")
    success = False
except TypeError as e:
    print("\n✅ REJEITADO como esperado!")
    print(f"   Erro: {e}")
    success = True
except Exception as e:
    print(f"\n⚠️ Erro inesperado: {type(e).__name__}: {e}")
    success = False

# =============================================================================
# TESTE 3: FlextResult[str].ok("hello") - DEVE PASSAR
# =============================================================================
print("\n" + "=" * 70)
print("TESTE 3: FlextResult[str].ok('hello') - tipo correto")
print("=" * 70)

try:
    result3 = FlextResultAutoValidate[str].ok("hello")
    print(f"\n✅ PASSOU: {result3._data}")
except Exception as e:
    print(f"\n❌ FALHOU: {type(e).__name__}: {e}")

# =============================================================================
# TESTE 4: FlextResult.ok(42) - sem tipo genérico
# =============================================================================
print("\n" + "=" * 70)
print("TESTE 4: FlextResult.ok(42) - SEM tipo genérico")
print("=" * 70)

try:
    result4 = FlextResultAutoValidate.ok(42)
    print(f"\n✅ PASSOU (sem validação): {result4._data}")
except Exception as e:
    print(f"\n❌ FALHOU: {type(e).__name__}: {e}")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "=" * 70)
print("RESUMO")
print("=" * 70)

if success:
    print("""
✅ FUNCIONA! get_args(cls) captura o tipo genérico!

DESCOBERTA:
- Quando fazemos FlextResult[int], Python cria um GenericAlias
- O GenericAlias tem __args__ = (int,)
- get_args() consegue extrair (int,) do GenericAlias
- Podemos validar com is_bearable(data, int)

RESULTADO:
✅ FlextResult[int].ok("string") É REJEITADO automaticamente!
✅ SEM precisar passar _type_hint explicitamente!
✅ API limpa e transparente!

PRÓXIMO PASSO:
Aplicar essa técnica ao FlextResult real.
    """)
else:
    print("""
❌ NÃO FUNCIONA
get_args() não conseguiu capturar o tipo genérico.
    """)
