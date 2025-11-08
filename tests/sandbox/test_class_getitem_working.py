"""Técnica DEFINITIVA: __class_getitem__ retornando subclasse.

OBJETIVO: FlextResult[int].ok("string") REJEITAR automaticamente.
"""

from typing import Generic, Self, TypeVar

from beartype import beartype
from beartype.door import is_bearable

T = TypeVar("T")

print("=" * 70)
print("TÉCNICA: __class_getitem__ com subclasse")
print("=" * 70)


@beartype
class FlextResultWithValidation(Generic[T]):
    """FlextResult que valida tipo genérico automaticamente."""

    _expected_type: type | None = None  # Armazenado na subclasse

    def __init__(self, data: T):
        # Validar tipo se _expected_type foi definido
        if self._expected_type is not None:
            if not is_bearable(data, self._expected_type):
                raise TypeError(
                    f"FlextResult[{self._expected_type.__name__}].ok() received "
                    f"{type(data).__name__} instead of {self._expected_type.__name__}"
                )

        self._data = data

    def __class_getitem__(cls, item):
        """Interceptar FlextResult[int] para criar subclasse com tipo.

        Quando fazem FlextResult[int], criamos uma subclasse que
        MANTÉM o tipo int armazenado.
        """
        print(f"  __class_getitem__ interceptado: FlextResult[{item}]")

        # Criar subclasse que mantém o tipo
        class TypedFlextResult(cls):
            _expected_type = item

        # Manter nome legível para debug
        TypedFlextResult.__name__ = f"{cls.__name__}[{getattr(item, '__name__', item)}]"

        return TypedFlextResult

    @classmethod
    def ok(cls, data: T) -> Self:
        """ok() usa _expected_type da subclasse para validar."""
        print(f"  ok() chamado em: {cls.__name__}")
        print(f"  _expected_type: {getattr(cls, '_expected_type', None)}")

        return cls(data)

    @property
    def value(self) -> T:
        return self._data


# =============================================================================
# TESTE 1: FlextResult[int].ok(42) - tipo CORRETO
# =============================================================================
print("\n" + "=" * 70)
print("TESTE 1: FlextResult[int].ok(42) - tipo correto")
print("=" * 70)

try:
    result1 = FlextResultWithValidation[int].ok(42)
    print(f"\n✅ PASSOU: {result1.value}")
except Exception as e:
    print(f"\n❌ FALHOU: {type(e).__name__}: {e}")

# =============================================================================
# TESTE 2: FlextResult[int].ok("string") - tipo ERRADO - DEVE REJEITAR!
# =============================================================================
print("\n" + "=" * 70)
print("TESTE 2: FlextResult[int].ok('string') - tipo ERRADO")
print("=" * 70)

try:
    result2 = FlextResultWithValidation[int].ok("string")
    print(f"\n❌ PASSOU (deveria rejeitar!): {result2.value}")
    success = False
except TypeError as e:
    print("\n✅ REJEITADO como esperado!")
    print(f"   Erro: {e}")
    success = True
except Exception as e:
    print(f"\n⚠️ Erro inesperado: {type(e).__name__}: {e}")
    success = False

# =============================================================================
# TESTE 3: FlextResult[str].ok("hello") - tipo CORRETO
# =============================================================================
print("\n" + "=" * 70)
print("TESTE 3: FlextResult[str].ok('hello') - tipo correto")
print("=" * 70)

try:
    result3 = FlextResultWithValidation[str].ok("hello")
    print(f"\n✅ PASSOU: {result3.value}")
except Exception as e:
    print(f"\n❌ FALHOU: {type(e).__name__}: {e}")

# =============================================================================
# TESTE 4: Sem tipo genérico
# =============================================================================
print("\n" + "=" * 70)
print("TESTE 4: FlextResult.ok(42) - SEM tipo genérico")
print("=" * 70)

try:
    result4 = FlextResultWithValidation.ok(42)
    print(f"\n✅ PASSOU (sem validação): {result4.value}")
except Exception as e:
    print(f"\n❌ FALHOU: {type(e).__name__}: {e}")

# =============================================================================
# TESTE 5: FlextResult[int].ok(42).value - propriedades funcionam?
# =============================================================================
print("\n" + "=" * 70)
print("TESTE 5: Propriedades e métodos funcionam?")
print("=" * 70)

try:
    result5 = FlextResultWithValidation[int].ok(99)
    value = result5.value
    print(f"\n✅ .value funciona: {value}")
except Exception as e:
    print(f"\n❌ FALHOU: {type(e).__name__}: {e}")

# =============================================================================
# RESUMO
# =============================================================================
print("\n" + "=" * 70)
print("RESUMO FINAL")
print("=" * 70)

if success:
    print("""
🎉 SUCESSO! __class_getitem__ FUNCIONA!

COMO FUNCIONA:
1. FlextResult[int] chama __class_getitem__(int)
2. Criamos subclasse TypedFlextResult com _expected_type = int
3. TypedFlextResult.ok() chama __init__ que valida com _expected_type
4. is_bearable(data, int) valida o tipo

RESULTADO:
✅ FlextResult[int].ok("string") É REJEITADO automaticamente!
✅ FlextResult[int].ok(42) PASSA normalmente!
✅ SEM precisar _type_hint explicitamente!
✅ API 100% transparente!

TRADE-OFFS:
✅ Validação automática de tipos genéricos
✅ API limpa (sem mudanças visíveis)
⚠️ Cria subclasses dinamicamente (overhead mínimo)
⚠️ Compatível com beartype

APLICAÇÃO AO FLEXT-CORE:
Esta técnica DEVE ser aplicada ao FlextResult real!
    """)
else:
    print("❌ FALHOU - técnica não funciona")
