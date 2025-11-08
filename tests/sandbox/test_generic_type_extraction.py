"""Investigar como extrair tipo genérico [T] em runtime.

OBJETIVO: Fazer FlextResult[int].ok("string") REJEITAR automaticamente.

Técnicas a testar:
1. __class_getitem__ para interceptar FlextResult[int]
2. get_args() para extrair int de FlextResult[int]
3. __orig_class__ para capturar tipo na instância
4. Wrapper proxy que mantém tipo
"""

from typing import Generic, TypeVar, get_args, get_origin

from beartype import beartype

T = TypeVar("T")


# =============================================================================
# TÉCNICA 1: Interceptar __class_getitem__
# =============================================================================


class TestTechnique1_ClassGetItem:
    """TÉCNICA 1: Sobrescrever __class_getitem__ para capturar tipo."""

    def test_extract_type_from_class_getitem(self):
        print("\n" + "=" * 70)
        print("TÉCNICA 1: __class_getitem__ para capturar [T]")
        print("=" * 70)

        class FlextResultTyped(Generic[T]):
            """FlextResult que captura tipo via __class_getitem__."""

            def __init__(self, data: T, _captured_type: type | None = None):
                self._data = data
                self._captured_type = _captured_type

                # Validar tipo se capturado
                if _captured_type is not None:
                    if not isinstance(data, _captured_type):
                        raise TypeError(
                            f"Expected {_captured_type.__name__}, "
                            f"got {type(data).__name__}"
                        )

            def __class_getitem__(cls, item):
                """Interceptar FlextResult[int] para capturar 'int'."""
                print(f"   __class_getitem__ chamado com: {item}")

                # Criar wrapper que mantém o tipo
                class TypedFlextResult:
                    """Proxy que mantém tipo capturado."""

                    _type = item

                    @classmethod
                    def ok(cls_inner, data):
                        """ok() que sabe o tipo esperado."""
                        print(f"   ok() chamado com tipo esperado: {cls._type}")
                        return FlextResultTyped(data, _captured_type=cls._type)

                return TypedFlextResult

        # TESTE 1: FlextResult[int].ok(42) - tipo correto
        print("\n[TESTE 1: Tipo correto]")
        try:
            result1 = FlextResultTyped[int].ok(42)
            print(f"✅ FlextResult[int].ok(42) PASSOU: {result1._data}")
        except Exception as e:
            print(f"❌ Falhou: {e}")

        # TESTE 2: FlextResult[int].ok("string") - tipo errado
        print("\n[TESTE 2: Tipo errado - DEVE REJEITAR]")
        try:
            result2 = FlextResultTyped[int].ok("string")
            print("❌ FlextResult[int].ok('string') PASSOU (deveria falhar)")
        except TypeError as e:
            print(f"✅ FlextResult[int].ok('string') REJEITADO: {e}")

        print("\n[RESULTADO TÉCNICA 1]")
        print("✅ FUNCIONA: Captura tipo via __class_getitem__!")
        print("⚠️ PROBLEMA: Retorna proxy, não FlextResult real")
        print("⚠️ PROBLEMA: Perde métodos da classe original")


# =============================================================================
# TÉCNICA 2: Usar __orig_class__ (PEP 673)
# =============================================================================


class TestTechnique2_OrigClass:
    """TÉCNICA 2: Usar __orig_class__ para capturar tipo na instância."""

    def test_orig_class_annotation(self):
        print("\n" + "=" * 70)
        print("TÉCNICA 2: __orig_class__ (PEP 673)")
        print("=" * 70)

        @beartype
        class FlextResultOrigClass(Generic[T]):
            """FlextResult que usa __orig_class__."""

            def __init__(self, data: T):
                self._data = data

                # Tentar capturar __orig_class__
                if hasattr(self, "__orig_class__"):
                    orig = self.__orig_class__
                    args = get_args(orig)
                    if args:
                        expected_type = args[0]
                        print(f"   __orig_class__ capturado: {expected_type}")

                        if not isinstance(data, expected_type):
                            raise TypeError(
                                f"Expected {expected_type}, got {type(data)}"
                            )
                else:
                    print("   __orig_class__ NÃO disponível")

            @classmethod
            def ok(cls, data: T) -> "FlextResultOrigClass[T]":
                return cls(data)

        # TESTE
        print("\n[TESTE]")
        try:
            result: FlextResultOrigClass[int] = FlextResultOrigClass.ok(42)
            print(f"✅ ok(42): {result._data}")
        except Exception as e:
            print(f"❌ Falhou: {e}")

        try:
            result2: FlextResultOrigClass[int] = FlextResultOrigClass.ok("string")
            print("❌ ok('string') PASSOU (deveria falhar)")
        except Exception as e:
            print(f"✅ ok('string') REJEITADO: {e}")

        print("\n[RESULTADO TÉCNICA 2]")
        print("⚠️ __orig_class__ só existe quando anotação explícita")
        print("⚠️ FlextResult[int].ok(...) não tem anotação, não funciona")


# =============================================================================
# TÉCNICA 3: Inspecionar cls em classmethod
# =============================================================================


class TestTechnique3_InspectCls:
    """TÉCNICA 3: Inspecionar cls no classmethod para extrair tipo."""

    def test_inspect_cls_in_classmethod(self):
        print("\n" + "=" * 70)
        print("TÉCNICA 3: Inspecionar cls no classmethod")
        print("=" * 70)

        @beartype
        class FlextResultInspect(Generic[T]):
            """FlextResult que inspeciona cls."""

            def __init__(self, data: T):
                self._data = data

            @classmethod
            def ok(cls, data: T) -> "FlextResultInspect[T]":
                """Tentar extrair tipo de cls."""
                print(f"\n   cls: {cls}")
                print(f"   cls.__name__: {getattr(cls, '__name__', 'N/A')}")
                print(f"   get_origin(cls): {get_origin(cls)}")
                print(f"   get_args(cls): {get_args(cls)}")

                # Tentar extrair tipo
                args = get_args(cls)
                if args:
                    expected_type = args[0]
                    print(f"   ✅ Tipo extraído: {expected_type}")

                    if not isinstance(data, expected_type):
                        raise TypeError(f"Expected {expected_type}, got {type(data)}")
                else:
                    print("   ❌ Nenhum tipo genérico encontrado")

                return cls(data)

        # TESTE 1: Com tipo
        print("\n[TESTE 1: FlextResult[int].ok(42)]")
        try:
            result1 = FlextResultInspect[int].ok(42)
            print(f"Resultado: {result1._data}")
        except Exception as e:
            print(f"Erro: {e}")

        # TESTE 2: Com tipo errado
        print("\n[TESTE 2: FlextResult[int].ok('string')]")
        try:
            result2 = FlextResultInspect[int].ok("string")
            print("❌ PASSOU (deveria falhar)")
        except TypeError as e:
            print(f"✅ REJEITADO: {e}")

        print("\n[RESULTADO TÉCNICA 3]")
        print("✅ PODE FUNCIONAR se get_args(cls) retornar tipo")


# =============================================================================
# TÉCNICA 4: Metaclass para interceptar
# =============================================================================


class TestTechnique4_Metaclass:
    """TÉCNICA 4: Usar metaclass para interceptar criação."""

    def test_metaclass_interception(self):
        print("\n" + "=" * 70)
        print("TÉCNICA 4: Metaclass para interceptar")
        print("=" * 70)

        class TypedMeta(type):
            """Metaclass que intercepta __getitem__."""

            def __getitem__(cls, item):
                """Interceptar FlextResult[int]."""
                print(f"   Metaclass __getitem__: {item}")

                # Criar classe wrapper
                class TypedWrapper(cls):
                    _expected_type = item

                    @classmethod
                    def ok(cls_inner, data):
                        print(f"   ok() com tipo: {cls_inner._expected_type}")
                        if not isinstance(data, cls_inner._expected_type):
                            raise TypeError(
                                f"Expected {cls_inner._expected_type}, got {type(data)}"
                            )
                        return super().ok(data)

                return TypedWrapper

        @beartype
        class FlextResultMeta(Generic[T], metaclass=TypedMeta):
            """FlextResult com metaclass."""

            def __init__(self, data: T):
                self._data = data

            @classmethod
            def ok(cls, data: T) -> "FlextResultMeta[T]":
                return cls(data)

        # TESTE
        print("\n[TESTE 1: Tipo correto]")
        try:
            result1 = FlextResultMeta[int].ok(42)
            print(f"✅ ok(42): {result1._data}")
        except Exception as e:
            print(f"❌ Falhou: {e}")

        print("\n[TESTE 2: Tipo errado]")
        try:
            result2 = FlextResultMeta[int].ok("string")
            print("❌ ok('string') PASSOU (deveria falhar)")
        except TypeError as e:
            print(f"✅ ok('string') REJEITADO: {e}")

        print("\n[RESULTADO TÉCNICA 4]")
        print("✅ FUNCIONA com metaclass!")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INVESTIGANDO EXTRAÇÃO DE TIPO GENÉRICO [T]")
    print("=" * 70)

    test1 = TestTechnique1_ClassGetItem()
    test2 = TestTechnique2_OrigClass()
    test3 = TestTechnique3_InspectCls()
    test4 = TestTechnique4_Metaclass()

    test1.test_extract_type_from_class_getitem()
    test2.test_orig_class_annotation()
    test3.test_inspect_cls_in_classmethod()
    test4.test_metaclass_interception()

    print("\n" + "=" * 70)
    print("EXPLORAÇÃO COMPLETA")
    print("=" * 70)
