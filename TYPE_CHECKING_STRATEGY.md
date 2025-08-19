# Estratégia de Verificação de Tipos - Flext Core

Este documento explica como o Pyright e Mypy trabalham em conjunto no projeto flext-core para maximizar a qualidade do código evitando redundâncias.

## Configuração da Estratégia

### Mypy (Strict Mode)

- **Responsabilidade**: Verificação principal de tipos, análise estática rigorosa
- **Configuração**: `mypy.ini` em modo strict máximo
- **Foco**: Correção de tipos, safety, verificações fundamentais

### Pyright (Complementar)

- **Responsabilidade**: Verificações específicas que o Pyright faz melhor
- **Configuração**: `pyrightconfig.json` em modo "basic" otimizado
- **Foco**: Performance, alcançabilidade, qualidade de código

## Divisão de Responsabilidades

### ✅ O que o MYPY verifica (desabilitado no Pyright)

- `reportGeneralTypeIssues` - Verificações básicas de tipo
- `reportOptionalMemberAccess` - Acesso a membros opcionais
- `reportOptionalCall` - Chamadas opcionais
- `reportMissingImports` - Imports ausentes
- `reportPossiblyUnboundVariable` - Variáveis possivelmente não definidas
- `reportIncompatibleVariableOverride` - Override incompatível de variáveis

### ✅ O que o PYRIGHT verifica (único responsável)

- `reportUnreachable` - Código inalcançável
- `reportUnusedImport` - Imports não utilizados
- `reportUnnecessaryCast` - Casts desnecessários
- `reportUnnecessaryComparison` - Comparações desnecessárias
- `reportInvalidTypeVarUse` - Uso inválido de TypeVar
- `reportIncompleteStub` - Stubs incompletos
- `reportUnawaited` - Awaits ausentes

### ⚖️ Verificações com nível reduzido (informational)

- `reportUnknownParameterType` - Tipos de parâmetros desconhecidos
- `reportUnknownVariableType` - Tipos de variáveis desconhecidos
- `reportUnusedFunction` - Funções não utilizadas
- `reportPrivateUsage` - Uso de membros privados

## Benefícios desta Abordagem

### 🚀 Performance

- Evita verificações duplicadas
- Pyright foca em análise incremental
- Mypy foca em verificação rigorosa de tipos

### 🎯 Especialização

- Cada ferramenta faz o que faz melhor
- Menor ruído de falsos positivos
- Diagnósticos mais precisos

### 🔧 Manutenibilidade

- Configuração clara e documentada
- Separação de responsabilidades
- Fácil ajuste de níveis de verificação

## Workflow Recomendado

1. **Durante desenvolvimento**: Pyright ativo no IDE para feedback imediato
2. **Pre-commit**: Mypy strict para verificação rigorosa
3. **CI/CD**: Ambos rodando em paralelo

## Comandos para Verificação

```bash
# Verificação completa com Mypy (strict)
mypy src/

# Verificação complementar com Pyright
pyright src/

# Verificação rápida apenas dos arquivos alterados
pyright --skipunannotated src/
```

## Configurações de IDE

### VS Code

```json
{
    "python.analysis.typeCheckingMode": "basic",
    "python.linting.mypyEnabled": false,
    "python.analysis.autoImportCompletions": true,
    "python.analysis.diagnosticMode": "workspace"
}
```

Esta configuração garante que o VS Code use o Pyright conforme nossa estratégia, sem conflitar com o Mypy que deve ser executado separadamente.
