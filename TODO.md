# FLEXT-CORE REFACTORING TODO

**Status**: Sistemática melhoria de qualidade em progresso  
**Objetivo**: 100% conformidade com padrões arquiteturais FLEXT conforme FLEXT_REFACTORING_PROMPT.md  
**Prioridade**: Alta - Base para todo ecossistema FLEXT (33 projetos dependentes)

---

## 🏆 FASES DE REFATORAÇÃO SISTEMÁTICA

### ✅ COMPLETADO - Arquitetura Base Estabelecida

- [x] FlextProtocols - Hierarquia de protocolos consolidada e completa
- [x] FlextCoreConstants - Sistema hierárquico de constantes implementado
- [x] FlextUtilities - Classe consolidada com classes aninhadas
- [x] FlextObservabilitySystem - Estrutura aninhada implementada
- [x] FlextDelegationSystem - Padrões de delegação consolidados

### 🚧 EM PROGRESSO - Classes Aninhadas Incompletas (PRIORIDADE MÁXIMA)

#### **1. FlextCoreModels - CRÍTICO**

**Arquivo**: `src/flext_core/models.py`  
**Status**: Parcialmente consolidado - precisa completar estruturas aninhadas  
**Problema**: Classes individuais (FlextModels, FlextModels.Entity, FlextModels.Value) existem, mas falta classe consolidada principal

**Ação Necessária**:

```python
class FlextCoreModels:
    """Consolidated class for all FLEXT model patterns."""

    # Existing classes become nested
    class Model(FlextModels): ...  # Atual FlextModels
    class Entity(FlextModels.Entity): ...  # Atual FlextModels.Entity
    class Value(FlextModels.Value): ...  # Atual FlextModels.Value
    class RootModel(FlextRootModel): ...  # Atual FlextRootModel

    # Factory methods consolidados
    class ModelFactory: ...
    class EntityFactory: ...
    class ValueObjectFactory: ...
```

**Compatibilidade**: Manter FlextModels, FlextModels.Entity, FlextModels.Value como aliases

#### **2. FlextCoreHandlers - CRÍTICO**

**Arquivo**: `src/flext_core/handlers.py`  
**Status**: Estrutura parcial existe, precisa completar consolidação  
**Problema**: FlextCoreHandlers existe mas classes aninhadas não estão completas

**Ação Necessária**:

```python
class FlextCoreHandlers:
    """Consolidated handler system - EXPANDIR classes aninhadas existentes"""

    # Completar estruturas existentes
    class AbstractHandler: ...  # ✅ Existe
    class Handler: ...  # ✅ Existe
    class ValidatingHandler: ...  # ✅ Existe
    class AuthorizingHandler: ...  # ✅ Existe
    class MetricsHandler: ...  # ✅ Existe

    # ADICIONAR classes faltantes:
    class CommandBus: ...  # ⚠️ Precisa mover para nested
    class QueryBus: ...  # ⚠️ Precisa mover para nested
    class HandlerChain: ...  # ⚠️ Precisa mover para nested
    class HandlerRegistry: ...  # ⚠️ Precisa mover para nested
    class Pipeline: ...  # ⚠️ Precisa mover para nested
```

#### **3. FlextCoreDecorators - CRÍTICO**

**Arquivo**: `src/flext_core/decorators.py`  
**Status**: FlextCoreDecorators existe, precisa validar completude  
**Problema**: Verificar se todas as classes de decoradores estão consolidadas

**Ação Necessária**: Auditoria completa das classes aninhadas e consolidação final

#### **4. FlextCoreValidation - ALTO**

**Arquivo**: `src/flext_core/validation.py`  
**Status**: Classes individuais existem, falta consolidação principal  
**Problema**: FlextValidations, FlextValidationPipeline, FlextDomainValidator como classes separadas

**Ação Necessária**:

```python
class FlextCoreValidation:
    """Consolidated validation system."""

    class Validation(FlextValidations): ...  # Classe principal
    class Pipeline(FlextValidationPipeline): ...
    class DomainValidator(FlextDomainValidator): ...
    class AbstractValidator(FlextAbstractValidator): ...

    # Predicates e validators consolidados
    class Predicates: ...
    class Validators: ...
```

#### **5. FlextCoreExceptions - ALTO**

**Arquivo**: `src/flext_core/exceptions.py`  
**Status**: FlextExceptions existe, verificar se está completo  
**Problema**: Validar se todas as exceções estão consolidadas sob uma classe principal

### 🔄 PRÓXIMAS FASES - Classes Menores

#### **6. FlextCoreFields - MÉDIO**

**Arquivo**: `src/flext_core/fields.py`  
**Status**: FlextFields existe, verificar estrutura aninhada

#### **7. FlextCoreMixins - MÉDIO**

**Arquivo**: `src/flext_core/mixins.py`  
**Status**: FlextMixins existe, verificar consolidação completa

#### **8. FlextCoreGuards - MÉDIO**

**Arquivo**: `src/flext_core/guards.py`  
**Status**: FlextGuards existe, validar estrutura

#### **9. FlextCoreLoggings - MÉDIO**

**Arquivo**: `src/flext_core.py`  
**Status**: FlextCoreLogging existe, verificar nome e estrutura

---

## 🔧 PADRÕES DE IMPLEMENTAÇÃO OBRIGATÓRIOS

### **Estrutura Consolidada Padrão**

```python
class FlextCore[Module]:
    """Single consolidated class containing ALL [module] functionality.

    Consolidates ALL [module] definitions into one class following FLEXT patterns.
    Individual classes available as nested classes for organization.
    """

    # Nested classes for organization
    class PrimaryClass: ...
    class SecondaryClass: ...

    # Legacy compatibility properties
    @property
    def LegacyClassName(self):
        return self.PrimaryClass
```

### **Compatibilidade Obrigatória**

- **SEMPRE**: Manter assinaturas antigas através de aliases
- **SEMPRE**: Manter importações existentes funcionando
- **SEMPRE**: Propriedades de compatibilidade para acesso legacy
- **NEVER**: Quebrar APIs públicas existentes

### **Exemplo de Migração Segura**

```python
# ANTES (individual)
class FlextModels(BaseModel): ...

# DEPOIS (consolidado + compatibilidade)
class FlextCoreModels:
    class Model(BaseModel): ...  # Implementação movida

    # Compatibility property
    @property
    def FlextModels(self):
        return self.Model

# Legacy alias (MANTER)
FlextModels = FlextCoreModels.Model  # OU FlextCoreModels().Model
```

---

## ⚡ AÇÕES IMEDIATAS (Esta Semana)

### **Prioridade 1: Auditoria Completa**

1. **Verificar FlextCoreHandlers**: Confirmar se todas as classes Handler estão nested
2. **Completar FlextCoreModels**: Implementar consolidação principal de models
3. **Validar FlextCoreDecorators**: Auditoria das classes aninhadas
4. **Revisar FlextCoreValidation**: Consolidar sistema de validação

### **Prioridade 2: Testes de Compatibilidade**

```bash
# Validar que imports legados funcionam
python -c "from flext_core import FlextModels, FlextModels.Entity, FlextResult"
python -c "from flext_core import FlextHandlerRegistry, FlextAbstractHandler"
python -c "from flext_core import FlextValidations, FlextValidators"

# Testar patterns consolidados
python -c "from flext_core import FlextCoreModels, FlextCoreHandlers"
```

### **Prioridade 3: Validação de Qualidade**

```bash
# DEPOIS de cada mudança - MANDATÓRIO
make validate  # Ruff + MyPy + Tests
ruff check src/flext_core --output-format=github
mypy src/flext_core --strict --show-error-codes
pytest tests/unit/core/ -v --tb=short
```

---

## 🎯 CRITÉRIOS DE SUCESSO

### **Validação Técnica**

- [ ] **0 erros de linting (Ruff)**
- [ ] **0 erros de type checking (MyPy strict)**
- [ ] **100% dos imports legados funcionando**
- [ ] **Todos os testes passando**

### **Validação Arquitetural**

- [ ] **Todas as classes principais consolidadas**
- [ ] **Estruturas aninhadas implementadas**
- [ ] **Compatibilidade legacy mantida**
- [ ] **Padrões SOLID seguidos**

### **Comando de Validação Final**

```bash
# Este comando DEVE retornar 100% sucesso
python -c "
from flext_core import FlextCoreModels, FlextCoreHandlers, FlextCoreDecorators
from flext_core import FlextCoreValidation, FlextCoreConstants, FlextProtocols
from flext_core import FlextModels, FlextModels.Entity, FlextResult  # Legacy
print('✅ ALL consolidated classes import successfully')
print('✅ ALL legacy aliases working')
print('🎯 FLEXT-CORE REFACTORING: COMPLETE')
"
```

---

## 🚨 RISCOS E MITIGAÇÕES

### **Risco**: Quebrar ecosystem dependente (33 projetos)

**Mitigação**: Manter 100% compatibilidade através de aliases

### **Risco**: Regressões durante refatoração

**Mitigação**: Executar `make validate` após CADA mudança

### **Risco**: Performance degradation

**Mitigação**: Benchmarks antes/depois, lazy loading onde necessário

---

## 📋 CHECKLIST DE EXECUÇÃO

### Para Cada Módulo

- [ ] 1. Ler código atual e entender estrutura
- [ ] 2. Identificar todas as classes que devem ser nested
- [ ] 3. Implementar classe consolidada principal
- [ ] 4. Mover classes existentes para nested
- [ ] 5. Implementar aliases de compatibilidade
- [ ] 6. Executar `make validate`
- [ ] 7. Testar imports legados
- [ ] 8. Executar testes específicos do módulo
- [ ] 9. Atualizar **all** exports
- [ ] 10. Marcar como concluído

---

**IMPORTANTE**: Este TODO deve ser executado com disciplina sistemática, um módulo por vez, validando completamente antes de prosseguir para o próximo.

**NEXT ACTION**: Começar com FlextCoreModels (maior impacto no ecosystem).
