"""System integration helper cases kept below the module LOC cap."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_tests import r

from tests.constants import c
from tests.typings import t
from tests.utilities import u

if TYPE_CHECKING:
    from tests.protocols import p


class TestsFlextFlextSystemWorkflowCases:
    def _test_complex_integration(self) -> None:
        """Test complex integration scenarios."""

        def processar_dados_usuario(
            dados: t.StrMapping,
        ) -> p.Result[t.StrMapping]:
            if not dados:
                return r[t.StrMapping].fail(
                    "Dados não fornecidos",
                    error_code=c.ErrorCode.VALIDATION_ERROR,
                )
            dados_processados: t.MutableStrMapping = {}
            for key, value in dados.items():
                if not u.string_non_empty(value):
                    return r[t.StrMapping].fail(
                        f"Campo '{key}' não pode estar vazio",
                        error_code=c.ErrorCode.VALIDATION_ERROR,
                    )
                dados_processados[key] = f"processado_{value}"
            dados_processados["processado_em"] = u.generate_iso_timestamp()
            dados_processados["processado_por"] = "sistema_flext"
            return r[t.StrMapping].ok(dados_processados)

        dados_teste = {"nome": "João", "email": "joao@exemplo.com"}
        resultado_processamento = processar_dados_usuario(dados_teste)
        assert resultado_processamento.success is True
        dados_finais = resultado_processamento.value
        assert "nome" in dados_finais
        assert "email" in dados_finais
        assert "processado_em" in dados_finais
        assert "processado_por" in dados_finais
        assert dados_finais["nome"] == "processado_João"
        assert dados_finais["email"] == "processado_joao@exemplo.com"
        dados_invalidos = {"nome": "", "email": "joao@exemplo.com"}
        resultado_erro = processar_dados_usuario(dados_invalidos)
        assert resultado_erro.success is False
        assert resultado_erro.error is not None
        assert "não pode estar vazio" in resultado_erro.error

    def _test_error_recovery(self) -> None:
        """Test error recovery scenarios."""
        resultado_com_erro: p.Result[str] = r[str].fail("erro_original")
        resultado_recuperado = resultado_com_erro.lash(
            lambda _error: r[str].ok("valor_recuperado"),
        )
        assert resultado_recuperado.success is True
        assert resultado_recuperado.value == "valor_recuperado"

        def operacao_1(data: str) -> p.Result[str]:
            return r[str].ok(f"etapa1_{data}")

        def operacao_2(data: str) -> p.Result[str]:
            if "erro" in data:
                return r[str].fail("erro_na_etapa2")
            return r[str].ok(f"etapa2_{data}")

        def operacao_3(data: str) -> p.Result[str]:
            return r[str].ok(f"final_{data}")

        pipeline_sucesso = (
            r[str]
            .ok("dados_iniciais")
            .flat_map(operacao_1)
            .flat_map(operacao_2)
            .flat_map(operacao_3)
        )
        assert pipeline_sucesso.success is True
        assert pipeline_sucesso.value == "final_etapa2_etapa1_dados_iniciais"
        pipeline_falha = (
            r[str]
            .ok("dados_com_erro")
            .flat_map(operacao_1)
            .flat_map(operacao_2)
            .flat_map(operacao_3)
        )
        assert pipeline_falha.success is False
        assert pipeline_falha.error == "erro_na_etapa2"
