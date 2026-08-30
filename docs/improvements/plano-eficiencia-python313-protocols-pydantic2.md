# Plano objetivo de eficiência: Python 3.13 (MRO), Protocols e Pydantic v2

## Objetivo

Definir **ações objetivas, priorizadas e mensuráveis** para reduzir custo de runtime no `flext-core`, mantendo tipagem
forte e segurança de contrato.

---

## Diagnóstico aprofundado (estado atual no código)

### A) Dispatcher usa `Protocol` runtime-checkable no hot path

- Em `src/flext_core/dispatcher.py`, `_execute_handler()` decide o caminho com:
  - `isinstance(handler, DispatchMessageProtocol)`
  - `isinstance(handler, HandleProtocol)`
  - `isinstance(handler, ExecuteProtocol)`
- Esses `Protocol` são `@runtime_checkable`, então cada `isinstance` é estrutural e mais caro que despacho por função pré-resolvida.

**Efeito prático**: custo repetido por mensagem no caminho crítico.

### B) Introspecção de protocolo com varredura de `mro()` sem cache

- Em `src/flext_core/protocols.py`, `_ProtocolIntrospection.validate_protocol_compliance()` percorre `target_cls.mro()`
  e anotações para cada validação.
- Em carga de módulos/classes, esse padrão escala mal quando há muitas subclasses/protocolos.

**Efeito prático**: piora de cold-start/import e custo de bootstrap.

### C) `TypeAdapter(...)` criado dentro de validação (repetição evitável)

- Em `src/flext_core/_models/cqrs.py`, `validate_pagination()` cria `TypeAdapter(...)` a cada chamada.
- Em `src/flext_core/_models/settings.py`, `BatchProcessingConfig.validate_batch()` também instancia adapter no fluxo.

**Efeito prático**: custo extra de construção de schema/adaptador em caminho frequente.

### D) Campos Pydantic com `default=[]` (coleções mutáveis)

- Há `Field(default=[])` em múltiplos modelos (`entity.py`, `service.py`, `generic.py`, `containers.py`, `settings.py`).

**Efeito prático**: além de risco semântico, aumenta chance de comportamentos inesperados e debugging mais caro.

### E) Validações e coerções que podem ser simplificadas com práticas atuais do Pydantic v2

- Já existe uso correto de `ConfigDict`, `field_validator`, `model_validator` e adapters em parte da base.
- Falta padronização para:
  - adapters cacheados por classe/módulo;
  - defaults mutáveis com `default_factory`;
  - minimizar validação redundante em loops internos.

---

## O que precisa ser feito (objetivo, por ordem)

## P0 — aplicar imediatamente (alto impacto / baixo risco)

1. **Trocar runtime protocol dispatch por função pré-compilada no registro**
   - Arquivo: `src/flext_core/dispatcher.py`.
   - Ação: no `register_handler()`, resolver uma vez o executor (`dispatch_message` / `handle` / `execute` / callable) e
     armazenar callable final.
   - Resultado esperado: `_execute_handler()` deixa de fazer cadeia de `isinstance(...Protocol)` por mensagem.
   - Critério de aceite: benchmark de dispatch com ganho de throughput e redução de p95.

2. **Cachear `TypeAdapter` em `ClassVar`/módulo nos validadores quentes**
   - Arquivos iniciais:
     - `src/flext_core/_models/cqrs.py` (`validate_pagination`)
     - `src/flext_core/_models/settings.py` (`validate_batch`)
   - Ação: mover adapters para constantes de classe/módulo (`ClassVar[TypeAdapter[...]]`).
   - Critério de aceite: zero criação dinâmica de adapter nesses métodos.

3. **Eliminar `Field(default=[])` em modelos Pydantic**
   - Arquivos alvo:
     - `src/flext_core/_models/entity.py`
     - `src/flext_core/_models/service.py`
     - `src/flext_core/_models/generic.py`
     - `src/flext_core/_models/containers.py`
     - `src/flext_core/_models/settings.py`
   - Ação: substituir por `Field(default_factory=list)`.
   - Critério de aceite: `rg "default=\[\]" src/flext_core/_models` sem ocorrências em modelos.

## P1 — estrutural (médio risco, alto retorno)

4. **Adicionar cache de conformidade de protocolo em `_ProtocolIntrospection`**
   - Arquivo: `src/flext_core/protocols.py`.
   - Ação:
     - cache para membros exigidos por protocolo;
     - cache para resultado de conformidade `(target_cls, protocol) -> bool`.
   - Observação: invalidar cache quando subclasses dinâmicas forem registradas.
   - Critério de aceite: redução mensurável de tempo em testes de bootstrap/import.

5. **Tornar validação profunda por metaclass configurável por ambiente**
   - Arquivo: `src/flext_core/protocols.py` (metaclass `ProtocolModelMeta`).
   - Ação: modo estrito em CI/dev; modo leve em produção.
   - Critério de aceite: cold-start melhor em produção sem perda de segurança em CI.

## P2 — governança e hardening contínuo

6. **Definir guideline oficial de Pydantic v2 para o projeto**
   - Documento interno com regras obrigatórias:
     - `default_factory` para coleções mutáveis;
     - `TypeAdapter` cacheado fora de loops/validators quentes;
     - validação estrita apenas na fronteira de entrada.

7. **Criar microbenchmarks e budget no CI**
   - Cenários mínimos:
     - dispatch com/sem resolução pré-compilada;
     - validação com adapter inline vs cacheado;
     - import/cold-start de módulos com protocolos/metaclass.
   - Budget sugerido inicial: regressão máxima de 5% em throughput e 10% em p95.

---

## Recomendações Pydantic v2 (atualizadas) para aplicar aqui

1. **Adapters reutilizáveis**: construir `TypeAdapter` uma vez e reutilizar.
2. **Coleções com `default_factory`**: evitar `default=[]` / `default={}`.
3. **Validação na borda**: usar validação mais estrita em input externo; evitar revalidar internamente sem necessidade.
4. **Evitar trabalho duplicado**: se o objeto já é `BaseModel` válido do tipo esperado, evitar roundtrip de
   validação/dump sem ganho funcional.
5. **Erros de validação agregados de forma estável**: padronizar construção de mensagens para manter custo previsível e
   facilitar profiling.

---

## Plano de execução em PRs pequenos (recomendado)

### PR 1 (rápido)

- `dispatcher.py`: resolver executor no `register_handler` e armazenar callable.
- `_models/cqrs.py`: cache do adapter de paginação.
- `_models/settings.py`: cache do adapter de batch.

### PR 2 (segurança + consistência)

- Remoção de `default=[]` em todos os modelos Pydantic alvo.
- Ajustes de testes de igualdade/default quando necessário.

### PR 3 (estrutura)

- Cache de conformidade de protocolos em `protocols.py`.
- Flag de ambiente para nível de validação da metaclass.

### PR 4 (controle de regressão)

- Benchmark scripts + budget de performance no CI.
- Documento de guideline oficial.

---

## Métricas esperadas

- **Dispatch throughput**: +15% a +30% (após PR 1).
- **Latência p95 de dispatch**: -10% a -25%.
- **Cold-start/import de módulos de protocolo**: -5% a -20% (após PR 3).
- **Estabilidade de validação/config**: redução de bugs ligados a defaults mutáveis.

---

## Definição objetiva de “pronto”

Considerar o ajuste concluído quando:

- Não houver `Field(default=[])` em modelos Pydantic de `src/flext_core/_models`.
- Dispatch não depender de cadeia de `isinstance(...Protocol)` por mensagem.
- Adapters quentes identificados estiverem cacheados.
- Benchmarks forem versionados e executados no CI com budget explícito.
