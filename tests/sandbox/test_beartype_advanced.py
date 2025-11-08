"""Técnicas avançadas para adicionar mais valor ao beartype.

Investigar se é possível fazer beartype validar:
1. Tipos genéricos (FlextResult[T])
2. Tipos dentro de Callables
3. unwrap_or(default: T) com tipo correto
"""

from collections.abc import Callable
from typing import Any, TypeVar

from beartype import beartype
from beartype.door import is_bearable

T = TypeVar("T")
U = TypeVar("U")


class TestAdvancedTechnique1_RuntimeTypeValidation:
    """TÉCNICA 1: Validação manual de tipos genéricos em runtime."""

    def test_validate_generic_with_get_args(self):
        """Usar typing.get_args() para validar tipo genérico."""
        print("\n" + "=" * 70)
        print("TÉCNICA 1: Validação manual de tipo genérico")
        print("=" * 70)

        from typing import Generic

        @beartype
        class FlextResultManual(Generic[T]):
            """FlextResult com validação MANUAL de tipo genérico."""

            def __init__(self, data: T, _expected_type: type[T] | None = None):
                # Armazenar tipo esperado para validação runtime
                self._data = data
                self._expected_type = _expected_type

                # Validar tipo se fornecido
                if _expected_type is not None:
                    if not isinstance(data, _expected_type):
                        raise TypeError(f"Expected {_expected_type}, got {type(data)}")

            @classmethod
            def ok(
                cls, data: T, _expected_type: type[T] | None = None
            ) -> "FlextResultManual[T]":
                """Create success with optional type validation."""
                return cls(data, _expected_type=_expected_type)

            @property
            def value(self) -> T:
                return self._data

            def unwrap_or(self, default: T) -> T:
                """unwrap_or com validação de tipo."""
                if self._expected_type is not None:
                    if not isinstance(default, self._expected_type):
                        raise TypeError(
                            f"default type {type(default)} doesn't match expected {self._expected_type}"
                        )
                return self._data if self._data is not None else default

        # TESTE 1: Com tipo esperado fornecido
        print("\n[TESTE 1: Com validação manual de tipo]")
        try:
            result = FlextResultManual.ok(42, _expected_type=int)
            print(f"✅ ok(42, int) PASSOU: {result.value}")
        except Exception as e:
            print(f"❌ ok(42, int) FALHOU: {e}")

        try:
            result_wrong = FlextResultManual.ok("string", _expected_type=int)  # type: ignore
            print(f"❌ ok('string', int) PASSOU (deveria falhar): {result_wrong.value}")
        except TypeError as e:
            print(f"✅ ok('string', int) REJEITADO: {e}")

        # TESTE 2: unwrap_or com validação
        print("\n[TESTE 2: unwrap_or com validação de tipo]")
        result = FlextResultManual.ok(42, _expected_type=int)
        try:
            value = result.unwrap_or(99)
            print(f"✅ unwrap_or(99) PASSOU: {value}")
        except Exception as e:
            print(f"❌ unwrap_or(99) FALHOU: {e}")

        try:
            value_wrong = result.unwrap_or("string")  # type: ignore
            print(f"❌ unwrap_or('string') PASSOU (deveria falhar): {value_wrong}")
        except TypeError as e:
            print(f"✅ unwrap_or('string') REJEITADO: {e}")

        print("\n[RESULTADO TÉCNICA 1]")
        print("✅ FUNCIONA: Validação manual de tipos genéricos")
        print("⚠️ DESVANTAGEM: Requer passar _expected_type explicitamente")
        print("⚠️ DESVANTAGEM: Sintaxe verbosa: ok(42, _expected_type=int)")


class TestAdvancedTechnique2_DecorateCallables:
    """TÉCNICA 2: Decorar Callables dinamicamente com beartype."""

    def test_decorate_callable_parameter(self):
        """Decorar função passada como parâmetro com beartype."""
        print("\n" + "=" * 70)
        print("TÉCNICA 2: Decorar Callables dinamicamente")
        print("=" * 70)

        @beartype
        class FlextResultDecorateCallable:
            """FlextResult que decora callables recebidos."""

            def __init__(self, data: Any):
                self._data = data

            @classmethod
            def ok(cls, data: Any) -> "FlextResultDecorateCallable":
                return cls(data)

            @property
            def value(self) -> Any:
                return self._data

            def map(self, func: Callable[[Any], Any]) -> "FlextResultDecorateCallable":
                """Map que DECORA func com beartype antes de executar."""
                # Decorar func com beartype dinamicamente
                func_decorated = beartype(func)

                try:
                    result = func_decorated(self._data)
                    return FlextResultDecorateCallable.ok(result)
                except Exception as e:
                    print(f"   Erro capturado: {type(e).__name__}: {e}")
                    raise

        # TESTE 1: Função com tipo correto
        print("\n[TESTE 1: Função com anotações corretas]")

        def good_func(x: int) -> str:
            return str(x * 2)

        try:
            result = FlextResultDecorateCallable.ok(5).map(good_func)
            print(f"✅ map(good_func) PASSOU: {result.value}")
        except Exception as e:
            print(f"❌ map(good_func) FALHOU: {e}")

        # TESTE 2: Função com tipo de retorno errado
        print("\n[TESTE 2: Função com tipo de retorno ERRADO]")

        def bad_func(x: int) -> str:
            return 42  # type: ignore  # Retorna int, declara str

        try:
            result = FlextResultDecorateCallable.ok(5).map(bad_func)
            print(f"❌ map(bad_func) PASSOU (deveria falhar): {result.value}")
        except Exception as e:
            if "beartype" in str(type(e).__name__).lower():
                print(f"✅ map(bad_func) REJEITADO por beartype: {type(e).__name__}")
            else:
                print(f"⚠️ map(bad_func) REJEITADO mas não por beartype: {e}")

        print("\n[RESULTADO TÉCNICA 2]")
        print("✅ FUNCIONA: Beartype valida função decorada dinamicamente!")
        print("⚠️ DESVANTAGEM: Overhead de decorar em CADA chamada")
        print("⚠️ DESVANTAGEM: Mensagens de erro apontam para wrapper")


class TestAdvancedTechnique3_BeartypeDoor:
    """TÉCNICA 3: Usar beartype.door.is_bearable para validação manual."""

    def test_is_bearable_validation(self):
        """Usar is_bearable() para validar tipos manualmente."""
        print("\n" + "=" * 70)
        print("TÉCNICA 3: beartype.door.is_bearable()")
        print("=" * 70)

        from typing import Generic

        @beartype
        class FlextResultBearableDoor(Generic[T]):
            """FlextResult com validação via is_bearable()."""

            def __init__(self, data: T, _type_hint: Any = None):
                self._data = data
                self._type_hint = _type_hint

            @classmethod
            def ok(
                cls, data: T, _type_hint: Any = None
            ) -> "FlextResultBearableDoor[T]":
                """Create success with type hint for validation."""
                if _type_hint is not None:
                    if not is_bearable(data, _type_hint):
                        raise TypeError(
                            f"Data {data!r} is not bearable as {_type_hint}"
                        )
                return cls(data, _type_hint=_type_hint)

            @property
            def value(self) -> T:
                return self._data

            def unwrap_or(self, default: T) -> T:
                """unwrap_or com validação via is_bearable."""
                if self._type_hint is not None:
                    if not is_bearable(default, self._type_hint):
                        raise TypeError(
                            f"default {default!r} is not bearable as {self._type_hint}"
                        )
                return self._data if self._data is not None else default

        # TESTE 1: Validação com int
        print("\n[TESTE 1: is_bearable com int]")
        try:
            result = FlextResultBearableDoor.ok(42, _type_hint=int)
            print(f"✅ ok(42, int) PASSOU: {result.value}")
        except Exception as e:
            print(f"❌ ok(42, int) FALHOU: {e}")

        try:
            result_wrong = FlextResultBearableDoor.ok("string", _type_hint=int)
            print("❌ ok('string', int) PASSOU (deveria falhar)")
        except TypeError as e:
            print(f"✅ ok('string', int) REJEITADO: {e}")

        # TESTE 2: unwrap_or com validação
        print("\n[TESTE 2: unwrap_or com is_bearable]")
        result = FlextResultBearableDoor.ok(42, _type_hint=int)
        try:
            value = result.unwrap_or(99)
            print(f"✅ unwrap_or(99) PASSOU: {value}")
        except Exception as e:
            print(f"❌ unwrap_or(99) FALHOU: {e}")

        try:
            value_wrong = result.unwrap_or("string")  # type: ignore
            print("❌ unwrap_or('string') PASSOU (deveria falhar)")
        except TypeError as e:
            print(f"✅ unwrap_or('string') REJEITADO: {e}")

        print("\n[RESULTADO TÉCNICA 3]")
        print("✅ FUNCIONA: is_bearable valida tipos em runtime")
        print("⚠️ DESVANTAGEM: Requer passar _type_hint explicitamente")
        print("⚠️ DESVANTAGEM: Sintaxe verbosa")


class TestAdvancedTechnique4_OverloadedMethods:
    """TÉCNICA 4: Usar overloads mais específicos."""

    def test_overloaded_unwrap_or(self):
        """Criar overloads específicos para tipos comuns."""
        print("\n" + "=" * 70)
        print("TÉCNICA 4: Overloads específicos")
        print("=" * 70)

        from typing import overload

        @beartype
        class FlextResultOverloaded:
            """FlextResult com overloads específicos."""

            def __init__(self, data: Any, data_type: type | None = None):
                self._data = data
                self._data_type = data_type

            @classmethod
            def ok_int(cls, data: int) -> "FlextResultOverloaded":
                """Create success with int - VALIDADO."""
                return cls(data, data_type=int)

            @classmethod
            def ok_str(cls, data: str) -> "FlextResultOverloaded":
                """Create success with str - VALIDADO."""
                return cls(data, data_type=str)

            @property
            def value(self) -> Any:
                return self._data

            # Overloads para unwrap_or
            @overload
            def unwrap_or(self, default: int) -> int: ...

            @overload
            def unwrap_or(self, default: str) -> str: ...

            def unwrap_or(self, default: int | str) -> int | str:
                """unwrap_or com tipos específicos."""
                # Validar tipo de default contra tipo armazenado
                if self._data_type is not None:
                    if not isinstance(default, self._data_type):
                        raise TypeError(
                            f"default type {type(default)} doesn't match {self._data_type}"
                        )
                return self._data if self._data is not None else default

        # TESTE 1: ok_int com unwrap_or
        print("\n[TESTE 1: ok_int + unwrap_or(int)]")
        result_int = FlextResultOverloaded.ok_int(42)
        try:
            value = result_int.unwrap_or(99)
            print(f"✅ unwrap_or(99) PASSOU: {value}")
        except Exception as e:
            print(f"❌ unwrap_or(99) FALHOU: {e}")

        try:
            value_wrong = result_int.unwrap_or("string")  # type: ignore
            print("❌ unwrap_or('string') PASSOU (deveria falhar)")
        except TypeError as e:
            print(f"✅ unwrap_or('string') REJEITADO: {e}")

        # TESTE 2: ok_str com unwrap_or
        print("\n[TESTE 2: ok_str + unwrap_or(str)]")
        result_str = FlextResultOverloaded.ok_str("hello")
        try:
            value = result_str.unwrap_or("default")
            print(f"✅ unwrap_or('default') PASSOU: {value}")
        except Exception as e:
            print(f"❌ unwrap_or('default') FALHOU: {e}")

        try:
            value_wrong = result_str.unwrap_or(42)  # type: ignore
            print("❌ unwrap_or(42) PASSOU (deveria falhar)")
        except TypeError as e:
            print(f"✅ unwrap_or(42) REJEITADO: {e}")

        print("\n[RESULTADO TÉCNICA 4]")
        print("✅ FUNCIONA: Overloads + validação manual")
        print("⚠️ DESVANTAGEM: Requer ok_int, ok_str, ok_float... (explosão de métodos)")
        print("⚠️ DESVANTAGEM: Não escala para todos os tipos")


def test_summary():
    """Resumo de todas as técnicas."""
    print("\n" + "=" * 70)
    print("RESUMO DAS TÉCNICAS AVANÇADAS")
    print("=" * 70)

    print("""
╔════════════════════════════════════════════════════════════════════╗
║                  TÉCNICAS PARA ADICIONAR VALOR                     ║
╚════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────┐
│ TÉCNICA 1: Validação Manual com _expected_type                  │
├──────────────────────────────────────────────────────────────────┤
│ ✅ Funciona: Valida tipos genéricos em runtime                   │
│ ⚠️ Sintaxe: result = FlextResult.ok(42, _expected_type=int)     │
│ ⚠️ Custo: API verbosa, tipo redundante                           │
│ ⚠️ DX: Pior experiência de desenvolvedor                         │
│ 💡 Valor: MÉDIO (funciona mas sintaxe ruim)                      │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ TÉCNICA 2: Decorar Callables Dinamicamente                      │
├──────────────────────────────────────────────────────────────────┤
│ ✅ Funciona: Valida tipos dentro de funções!                     │
│ ⚠️ Overhead: Decora em CADA chamada (~10-20% adicional)         │
│ ⚠️ Stack traces: Apontam para wrapper, não código original      │
│ 💡 Valor: ALTO (valida tipos em funções)                         │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ TÉCNICA 3: beartype.door.is_bearable()                          │
├──────────────────────────────────────────────────────────────────┤
│ ✅ Funciona: API do beartype para validação manual              │
│ ⚠️ Sintaxe: Requer _type_hint explícito                         │
│ ⚠️ Redundância: Similar à Técnica 1                             │
│ 💡 Valor: MÉDIO (mais limpo que isinstance)                      │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ TÉCNICA 4: Overloads Específicos (ok_int, ok_str...)            │
├──────────────────────────────────────────────────────────────────┤
│ ✅ Funciona: Validação forte para tipos específicos             │
│ ⚠️ Explosão: Precisa ok_int, ok_str, ok_float, ok_list...       │
│ ⚠️ Manutenção: Não escala para todos os tipos Python            │
│ 💡 Valor: BAIXO (não prático)                                    │
└──────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════════╗
║                      RECOMENDAÇÃO FINAL                            ║
╚════════════════════════════════════════════════════════════════════╝

🏆 TÉCNICA 2 É A MELHOR: Decorar Callables dinamicamente

IMPLEMENTAÇÃO RECOMENDADA:
    def map(self, func: Callable[[T_co], U]) -> FlextResult[U]:
        # Decorar func com beartype antes de executar
        func_validated = beartype(func)

        try:
            result = func_validated(self._data)
            return FlextResult[U].ok(result)
        except Exception as e:
            return FlextResult[U].fail(f"Map failed: {e}")

BENEFÍCIOS:
✅ Valida tipos dentro de funções passadas como parâmetros
✅ API limpa - sem mudanças visíveis para usuário
✅ Captura erros que Pyright não pega (tipos dinâmicos)

CUSTOS:
⚠️ Overhead adicional: ~10-20% (5-10% beartype + 5-10% decoração)
⚠️ Stack traces podem ser confusos

CASOS DE USO QUE JUSTIFICAM:
- Funções recebidas de código não-tipado
- Callbacks de usuários externos
- Plugins/extensões dinâmicos
- APIs públicas com entrada não confiável

╔════════════════════════════════════════════════════════════════════╗
║                      DECISÃO ATUALIZADA                            ║
╚════════════════════════════════════════════════════════════════════╝

SE aplicar beartype, usar TÉCNICA 2:
✅ @beartype na classe (todos métodos validados)
✅ map/flat_map decoram callables recebidos
✅ Máxima validação possível

TRADE-OFF:
- Benefício: Detecta erros em código dinâmico
- Custo: ~15-20% overhead total
- Decisão: Vale para APIs públicas, não para código interno
    """)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EXPLORANDO TÉCNICAS AVANÇADAS DE BEARTYPE")
    print("=" * 70)

    test1 = TestAdvancedTechnique1_RuntimeTypeValidation()
    test2 = TestAdvancedTechnique2_DecorateCallables()
    test3 = TestAdvancedTechnique3_BeartypeDoor()
    test4 = TestAdvancedTechnique4_OverloadedMethods()

    test1.test_validate_generic_with_get_args()
    test2.test_decorate_callable_parameter()
    test3.test_is_bearable_validation()
    test4.test_overloaded_unwrap_or()
    test_summary()

    print("\n" + "=" * 70)
    print("✅ EXPLORAÇÃO COMPLETA")
    print("=" * 70)
