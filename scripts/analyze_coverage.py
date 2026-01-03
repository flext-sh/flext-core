#!/usr/bin/env python3
"""Análise de cobertura linha-por-linha para consolidação de testes.

Gera relatório detalhado mostrando:
1. Quais testes cobrem quais linhas de cada módulo
2. Linhas não cobertas por módulo
3. Testes redundantes (que cobrem as mesmas linhas)
4. Recomendações de consolidação
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from coverage import CoverageData


@dataclass(frozen=True, slots=True)
class LineInfo:
    """Informação sobre uma linha de código."""

    file_path: str
    line_number: int
    covered_by_tests: list[str]
    is_covered: bool


@dataclass(frozen=True, slots=True)
class ModuleCoverageAnalysis:
    """Análise de cobertura de um módulo."""

    module_path: str
    total_lines: int
    covered_lines: int
    missing_lines: int
    coverage_percent: float
    line_mapping: dict[int, list[str]]  # linha → testes
    tests_covering_module: set[str]


class CoverageAnalyzer:
    """Analisador de cobertura linha-por-linha."""

    def __init__(self, coverage_file: str = ".coverage") -> None:
        """Initialize the coverage analyzer.

        Args:
            coverage_file: Path to the coverage data file (default: .coverage)

        """
        super().__init__()
        self.cov_data = CoverageData(basename=coverage_file)
        self.cov_data.read()

    def analyze_module(self, module_path: str) -> ModuleCoverageAnalysis:
        """Analisa cobertura de um módulo específico."""
        # Obter linhas executadas
        executed_lines_raw = self.cov_data.lines(module_path)
        executed_lines = set(executed_lines_raw) if executed_lines_raw else set()

        # Obter mapeamento linha → contextos (testes)
        contexts_map = self.cov_data.contexts_by_lineno(module_path)

        # Coletar todos os testes que tocam este módulo
        all_tests: set[str] = set()
        for tests in contexts_map.values():
            all_tests.update(test.split("|")[0] for test in tests)

        # Calcular métricas
        missing_lines = self._get_missing_lines(module_path)
        total_lines = len(executed_lines) + len(missing_lines)
        coverage_percent = (
            (len(executed_lines) / total_lines * 100) if total_lines > 0 else 0
        )

        return ModuleCoverageAnalysis(
            module_path=module_path,
            total_lines=total_lines,
            covered_lines=len(executed_lines),
            missing_lines=len(missing_lines),
            coverage_percent=coverage_percent,
            line_mapping=contexts_map,
            tests_covering_module=all_tests,
        )

    def _get_missing_lines(self, module_path: str) -> set[int]:
        """Obtém linhas não cobertas via JSON report."""
        json_path = Path("coverage.json")
        if not json_path.exists():
            return set()

        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)

        file_data = data.get("files", {}).get(module_path, {})
        return set(file_data.get("missing_lines", []))

    def find_redundant_tests(self, module_path: str) -> dict[frozenset[int], list[str]]:
        """Identifica testes que cobrem exatamente as mesmas linhas."""
        analysis = self.analyze_module(module_path)

        # Mapear teste → linhas cobertas
        test_to_lines: dict[str, set[int]] = defaultdict(set)
        for line, tests in analysis.line_mapping.items():
            for test in tests:
                test_name = test.split("|")[0]
                test_to_lines[test_name].add(line)

        # Agrupar testes por conjunto de linhas cobertas
        lines_to_tests: dict[frozenset[int], list[str]] = defaultdict(list)
        for test, lines in test_to_lines.items():
            lines_to_tests[frozenset(lines)].append(test)

        # Retornar apenas grupos com 2+ testes (redundantes)
        return {
            lines: tests for lines, tests in lines_to_tests.items() if len(tests) > 1
        }

    def generate_consolidation_report(self, target_modules: list[str]) -> str:
        """Gera relatório de consolidação para múltiplos módulos."""
        report_lines = [
            "# RELATÓRIO DE ANÁLISE DE COBERTURA E CONSOLIDAÇÃO",
            "",
            f"Analisando {len(target_modules)} módulos críticos",
            "=" * 80,
            "",
        ]

        for module in target_modules:
            analysis = self.analyze_module(module)
            redundant = self.find_redundant_tests(module)

            report_lines.extend([
                f"\n## {module}",
                f"Cobertura: {analysis.coverage_percent:.2f}%",
                f"Linhas totais: {analysis.total_lines}",
                f"Linhas cobertas: {analysis.covered_lines}",
                f"Linhas NÃO cobertas: {analysis.missing_lines}",
                f"Testes que tocam este módulo: {len(analysis.tests_covering_module)}",
                "",
            ])

            # Linhas não cobertas
            missing = self._get_missing_lines(module)
            if missing:
                report_lines.extend([
                    "### Linhas NÃO COBERTAS:",
                    f"  {sorted(missing)[:50]}{'...' if len(missing) > 50 else ''}",
                    "",
                ])

            # Testes redundantes
            if redundant:
                report_lines.extend([
                    "### TESTES REDUNDANTES (cobrem mesmas linhas):",
                    "",
                ])
                for idx, (lines, tests) in enumerate(list(redundant.items())[:10], 1):
                    report_lines.append(
                        f"  Grupo {idx}: {len(tests)} testes cobrem {len(lines)} linhas"
                    )
                    for test in tests[:3]:  # Primeiros 3 testes
                        test_name = test.split("|")[0]
                        report_lines.append(f"    - {test_name}")
                    if len(tests) > 3:
                        report_lines.append(f"      ... e mais {len(tests) - 3} testes")
                    report_lines.append("")

            # Mapeamento linha → testes (primeiras 10 linhas)
            report_lines.extend([
                "### MAPEAMENTO (primeiras 10 linhas cobertas):",
                "",
            ])
            for line_num in sorted(list(analysis.line_mapping.keys())[:10]):
                tests = analysis.line_mapping[line_num]
                report_lines.append(f"  Linha {line_num}: {len(tests)} teste(s)")
                for test in tests[:3]:  # Primeiros 3 testes
                    test_name = test.split("|")[0]
                    report_lines.append(f"      → {test_name}")
                if len(tests) > 3:
                    report_lines.append(f"      ... e mais {len(tests) - 3} testes")

            report_lines.append("")

        return "\n".join(report_lines)


def main() -> None:
    """Executa análise de cobertura."""
    # Módulos críticos com baixa cobertura
    critical_modules = [
        "src/flext_core/_utilities/validation.py",
        "src/flext_core/_utilities/parser.py",
        "src/flext_core/_utilities/reliability.py",
        "src/flext_core/registry.py",
    ]

    analyzer = CoverageAnalyzer()
    report = analyzer.generate_consolidation_report(critical_modules)

    # Salvar relatório
    output_path = Path("COVERAGE_LINE_BY_LINE_ANALYSIS.md")
    _ = output_path.write_text(report, encoding="utf-8")

    print(f"✅ Relatório gerado: {output_path}")
    print(report)


if __name__ == "__main__":
    main()
