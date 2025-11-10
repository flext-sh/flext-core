"""Teste mínimo para validar @beartype com Self type hints.

OBJETIVO: Descobrir se @beartype funciona com Self nos construtores.
"""

import pytest
from beartype.roar import BeartypeDecorHintPep673Exception

from tests.sandbox.flext_result_beartype_minimal import FlextResultBeartype


class TestBeartypeWithSelf:
    """Fase 1: Testar se @beartype funciona com Self."""

    def test_ok_basic_usage(self):
        """TESTE 1: ok() com @beartype - caso básico."""
        print("\n=== TESTE 1: ok() com @beartype ===")
        try:
            result = FlextResultBeartype[int].ok(42)
            assert result.is_success
            assert result.value == 42
            print("✅ SUCCESS: ok() funciona com @beartype!")
        except BeartypeDecorHintPep673Exception as e:
            print(f"❌ FAIL: ok() não suporta @beartype: {e}")
            pytest.fail(f"ok() com @beartype falhou: {e}")
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
            pytest.fail(f"Erro inesperado: {e}")

    def test_fail_basic_usage(self):
        """TESTE 2: fail() com @beartype - caso básico."""
        print("\n=== TESTE 2: fail() com @beartype ===")
        try:
            result = FlextResultBeartype[int].fail("Test error")
            assert result.is_failure
            assert result.error == "Test error"
            print("✅ SUCCESS: fail() funciona com @beartype!")
        except BeartypeDecorHintPep673Exception as e:
            print(f"❌ FAIL: fail() não suporta @beartype: {e}")
            pytest.fail(f"fail() com @beartype falhou: {e}")
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
            pytest.fail(f"Erro inesperado: {e}")

    def test_map_basic_usage(self):
        """TESTE 3: map() com @beartype - deve funcionar."""
        print("\n=== TESTE 3: map() com @beartype ===")
        try:
            result = FlextResultBeartype[int].ok(5).map(lambda x: x * 2)
            assert result.is_success
            assert result.value == 10
            print("✅ SUCCESS: map() funciona com @beartype!")
        except Exception as e:
            print(f"❌ FAIL: map() falhou: {type(e).__name__}: {e}")
            pytest.fail(f"map() com @beartype falhou: {e}")

    def test_flat_map_basic_usage(self):
        """TESTE 4: flat_map() com @beartype - deve funcionar."""
        print("\n=== TESTE 4: flat_map() com @beartype ===")
        try:
            result = (
                FlextResultBeartype[int]
                .ok(5)
                .flat_map(lambda x: FlextResultBeartype[int].ok(x * 2))
            )
            assert result.is_success
            assert result.value == 10
            print("✅ SUCCESS: flat_map() funciona com @beartype!")
        except Exception as e:
            print(f"❌ FAIL: flat_map() falhou: {type(e).__name__}: {e}")
            pytest.fail(f"flat_map() com @beartype falhou: {e}")

    def test_unwrap_basic_usage(self):
        """TESTE 5: unwrap() com @beartype - deve funcionar."""
        print("\n=== TESTE 5: unwrap() com @beartype ===")
        try:
            result = FlextResultBeartype[int].ok(42)
            value = result.unwrap()
            assert value == 42
            print("✅ SUCCESS: unwrap() funciona com @beartype!")
        except Exception as e:
            print(f"❌ FAIL: unwrap() falhou: {type(e).__name__}: {e}")
            pytest.fail(f"unwrap() com @beartype falhou: {e}")

    def test_railway_pattern(self):
        """TESTE 6: Railway pattern completo com @beartype."""
        print("\n=== TESTE 6: Railway pattern com @beartype ===")
        try:
            result = (
                FlextResultBeartype[int]
                .ok(5)
                .map(lambda x: x * 2)
                .flat_map(lambda x: FlextResultBeartype[int].ok(x + 1))
                .unwrap_or(0)
            )
            assert result == 11
            print("✅ SUCCESS: Railway pattern funciona com @beartype!")
        except Exception as e:
            print(f"❌ FAIL: Railway pattern falhou: {type(e).__name__}: {e}")
            pytest.fail(f"Railway pattern com @beartype falhou: {e}")


class TestBeartypeTypeValidation:
    """Fase 2: Testar se @beartype valida tipos."""

    def test_map_catches_wrong_return_type(self):
        """TESTE 7: @beartype deve capturar tipo de retorno errado em map()."""
        print("\n=== TESTE 7: Validação de tipo em map() ===")

        result = FlextResultBeartype[int].ok(5)

        def bad_func(x: int) -> str:
            return 42

        try:
            result.map(bad_func)
            print(
                "⚠️ WARNING: map() NÃO capturou tipo errado (beartype pode não validar isso)"
            )
        except Exception as e:
            if "beartype" in str(type(e).__name__).lower():
                print(f"✅ SUCCESS: @beartype capturou tipo errado: {e}")
            else:
                print(f"⚠️ Erro capturado mas não por beartype: {type(e).__name__}")


if __name__ == "__main__":
    # Executar testes diretamente
    import sys

    test_class = TestBeartypeWithSelf()
    test_validation = TestBeartypeTypeValidation()

    tests = [
        ("ok_basic_usage", test_class.test_ok_basic_usage),
        ("fail_basic_usage", test_class.test_fail_basic_usage),
        ("map_basic_usage", test_class.test_map_basic_usage),
        ("flat_map_basic_usage", test_class.test_flat_map_basic_usage),
        ("unwrap_basic_usage", test_class.test_unwrap_basic_usage),
        ("railway_pattern", test_class.test_railway_pattern),
        ("map_type_validation", test_validation.test_map_catches_wrong_return_type),
    ]

    print("\n" + "=" * 70)
    print("EXECUTANDO TESTES MÍNIMOS DE BEARTYPE")
    print("=" * 70)

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ TESTE {name} FALHOU: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTADOS: {passed} passed, {failed} failed")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
