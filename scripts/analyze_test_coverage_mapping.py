#!/usr/bin/env python3
"""Análise: Quais testes testam quais módulos - Consolidação de testes.

RESPONDE A PERGUNTA CRÍTICA:
1. Quais testes testam validation.py?
2. Quais testes testam parser.py?
3. Quais testes testam reliability.py?
4. Quais testes testam registry.py?

Gera MAPA DE CONSOLIDAÇÃO:
- Para cada módulo crítico, lista TODOS os arquivos de teste que o testam
- Identifica testes REDUNDANTES (que testam exatamente as mesmas coisas)
- Recomenda consolidação em arquivo único por módulo

Execução:
  python scripts/analyze_test_coverage_mapping.py
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModuleTestMapping:
    """Mapa de testes para um módulo."""

    module_name: str
    module_path: str
    test_files: list[str]  # Arquivos de teste que testam este módulo
    test_count_by_file: dict[str, int]  # { arquivo.py: N_testes }
    imports_in_test_files: dict[str, list[str]]  # { arquivo.py: [imports] }
    total_test_files: int
    estimated_redundancy: float  # 0-1 (1 = completamente redundante)


class TestCoverageMapper:
    """Mapeia testes para módulos testados."""

    def __init__(self) -> None:
        """Initialize the test coverage mapper with project paths."""
        super().__init__()
        self.project_root = Path(__file__).resolve().parent.parent
        self.tests_dir = self.project_root / "tests"
        self.src_dir = self.project_root / "src"

    def find_test_files_importing_module(self, module_path: str) -> list[str]:
        """Encontra TODOS os arquivos de teste que importam um módulo.

        Args:
            module_path: ex. "src/flext_core/_utilities/validation.py"

        Returns:
            Lista de caminhos de arquivos de teste que importam este módulo

        """
        # Extrair nome do módulo para grep
        module_name = Path(module_path).stem  # validation, parser, etc.

        # Procurar por importações do módulo em todos os testes
        cmd = [
            "grep",
            "-r",
            f"from flext_core.*{module_name} import|import.*{module_name}",
            str(self.tests_dir),
            "--include=*.py",
        ]

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                files = set()
                for line in result.stdout.strip().split("\n"):
                    if line:
                        file_path = line.split(":")[0]
                        files.add(file_path)
                return sorted(files)
            return []
        except subprocess.TimeoutExpired:
            return []
        except Exception:
            return []

    def count_tests_in_file(self, test_file: str) -> int:
        """Conta quantos testes (def test_) estão em um arquivo.

        Args:
            test_file: Caminho do arquivo de teste

        Returns:
            Número de funções de teste

        """
        cmd = ["grep", "-c", "^def test_", test_file]

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
            return 0
        except Exception:
            return 0

    def extract_imports_from_file(self, test_file: str) -> list[str]:
        """Extrai linhas de import/from de um arquivo.

        Args:
            test_file: Caminho do arquivo de teste

        Returns:
            Lista de linhas que contêm imports

        """
        cmd = ["grep", "-E", "^from|^import", test_file]

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")
            return []
        except Exception:
            return []

    def estimate_redundancy(
        self, test_files: list[str], imports_map: dict[str, list[str]]
    ) -> float:
        """Estima grau de redundância entre arquivos de teste.

        Quanto maior a similaridade entre imports, mais redundância há.

        Args:
            test_files: Lista de arquivos de teste
            imports_map: Mapa de arquivos para suas importações

        Returns:
            Valor 0-1 indicando redundância (1 = completamente redundante)

        """
        if len(test_files) <= 1:
            return 0.0

        # Converter imports para sets de nomes de módulos testados
        module_sets = []
        for file in test_files:
            imports = imports_map.get(file, [])
            # Extrair nomes de módulos dos imports
            module_names = set()
            for imp in imports:
                if "import" in imp:
                    # Simplificar: extrair primeira palavra após import/from
                    parts = imp.split()
                    if "from" in parts:
                        idx = parts.index("from")
                        if idx + 1 < len(parts):
                            module_names.add(parts[idx + 1])
                    elif "import" in parts:
                        idx = parts.index("import")
                        if idx + 1 < len(parts):
                            module_names.add(parts[idx + 1])
            module_sets.append(module_names)

        # Calcular similaridade pairwise
        if not module_sets:
            return 0.0

        similarities = []
        for i in range(len(module_sets)):
            for j in range(i + 1, len(module_sets)):
                set1, set2 = module_sets[i], module_sets[j]
                if set1 and set2:
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    if union > 0:
                        jaccard = intersection / union
                        similarities.append(jaccard)

        if similarities:
            return sum(similarities) / len(similarities)
        return 0.0

    def analyze_module(self, module_path: str, module_name: str) -> ModuleTestMapping:
        """Analisa cobertura de teste para um módulo específico.

        Args:
            module_path: ex. "src/flext_core/_utilities/validation.py"
            module_name: ex. "validation"

        Returns:
            ModuleTestMapping com análise completa

        """
        test_files = self.find_test_files_importing_module(module_path)

        test_count_by_file = {}
        imports_by_file = {}

        for test_file in test_files:
            count = self.count_tests_in_file(test_file)
            test_count_by_file[test_file] = count
            imports_by_file[test_file] = self.extract_imports_from_file(test_file)

        redundancy = self.estimate_redundancy(test_files, imports_by_file)

        return ModuleTestMapping(
            module_name=module_name,
            module_path=module_path,
            test_files=test_files,
            test_count_by_file=test_count_by_file,
            imports_in_test_files=imports_by_file,
            total_test_files=len(test_files),
            estimated_redundancy=redundancy,
        )

    def generate_consolidation_report(self) -> str:
        """Gera relatório de consolidação: QUAIS TESTES TESTAM QUAIS MÓDULOS.

        Returns:
            Texto do relatório formatado

        """
        modules = [
            ("validation.py", "src/flext_core/_utilities/validation.py"),
            ("parser.py", "src/flext_core/_utilities/parser.py"),
            ("reliability.py", "src/flext_core/_utilities/reliability.py"),
            ("registry.py", "src/flext_core/registry.py"),
        ]

        lines = [
            "# MAPA DE CONSOLIDAÇÃO: QUAIS TESTES TESTAM QUAIS MÓDULOS",
            "",
            "Responde à pergunta crítica: Para cada módulo, quais testes o testam?",
            "Identifica redundância e recomenda consolidação em arquivo único.",
            "",
            "=" * 80,
            "",
        ]

        for module_name, module_path in modules:
            analysis = self.analyze_module(module_path, module_name)

            lines.extend([
                f"## {analysis.module_name}",
                f"Módulo: {analysis.module_path}",
                f"Total de arquivos de teste: {analysis.total_test_files}",
                f"Redundância estimada: {analysis.estimated_redundancy * 100:.1f}%",
                "",
                "### Arquivos de teste que testam este módulo:",
                "",
            ])

            if analysis.test_files:
                total_tests = 0
                for test_file in analysis.test_files:
                    count = analysis.test_count_by_file.get(test_file, 0)
                    total_tests += count
                    relative_path = test_file.replace(str(self.project_root), "")
                    lines.append(f"- **{relative_path}** ({count} testes)")

                lines.extend([
                    "",
                    f"**Total de testes para {analysis.module_name}: {total_tests} testes**",
                    "",
                ])

                # Mostrar imports em cada arquivo
                lines.extend(("### Imports em cada arquivo:", ""))
                for test_file in analysis.test_files:
                    relative_path = test_file.replace(str(self.project_root), "")
                    imports = analysis.imports_in_test_files.get(test_file, [])
                    lines.append(f"#### {relative_path}")
                    if imports:
                        # Primeiros 5 imports
                        lines.extend(f"  {imp.strip()}" for imp in imports[:5])
                        if len(imports) > 5:
                            lines.append(f"  ... e mais {len(imports) - 5} imports")
                    lines.append("")

                # Recomendação de consolidação
                if analysis.total_test_files > 1:
                    lines.extend([
                        "### 🎯 RECOMENDAÇÃO DE CONSOLIDAÇÃO:",
                        "",
                        f"**CONSOLIDAR em**: `tests/unit/test_utilities_{analysis.module_name.replace('.py', '')}.py`",
                        "",
                        "**Estratégia**:",
                        "1. ✅ Centralizar TODOS os cenários em `tests/helpers/scenarios.py`",
                        f"2. ✅ Criar arquivo único: `test_utilities_{analysis.module_name.replace('.py', '')}.py`",
                        "3. ✅ Usar `@pytest.mark.parametrize` com cenários centralizados",
                        f"4. ✅ Backup e remover {analysis.total_test_files} arquivos antigos",
                        "5. ✅ Validar 100% de cobertura no novo arquivo consolidado",
                        "",
                        f"**Estimativa de redução**: ~60-70% menos linhas de teste (de {sum(analysis.test_count_by_file.values())} para ~30-40% disso)",
                        "",
                    ])
            else:
                lines.extend((
                    "⚠️ Nenhum arquivo de teste encontrado importando este módulo!",
                    "",
                ))

            lines.extend(("-" * 80, ""))

        return "\n".join(lines)


def main() -> None:
    """Executa análise de mapeamento testes → módulos."""
    mapper = TestCoverageMapper()
    report = mapper.generate_consolidation_report()

    # Salvar relatório
    output_path = Path(__file__).resolve().parent.parent / "TEST_COVERAGE_MAPPING.md"
    _ = output_path.write_text(report)

    print(f"✅ Mapa de consolidação gerado: {output_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
