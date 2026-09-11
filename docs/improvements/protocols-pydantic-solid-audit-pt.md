# Auditoria: Protocols + Pydantic + SOLID (FLEXT)

## Contexto

Este documento identifica _bad patterns_ observados no uso de `Protocol` e na integração com Pydantic no projeto,
seguido de práticas recomendadas (“Pydantic way” + princípios SOLID).

## Bad patterns identificados

1. **Dependência de APIs internas/privadas para metaclass de Protocol + Pydantic**
   - `typing._ProtocolMeta` e `pydantic._internal._model_construction.ModelMetaclass` são imports internos e frágeis a
     mudanças de versão.
   - Risco: quebra silenciosa em upgrades de Python/Pydantic.

2. **Detecção e validação manual de Protocol por introspecção frágil**
   - Uso de atributos internos como `__protocol_attrs__` e lógica heurística para inferir protocol em runtime.
   - Risco: comportamento inconsistente entre versões e branchs de tipagem.

3. **Protocol de configuração genérico demais, mas consumidores exigem campos extras via `getattr`**
   - `p.Config` define poucos campos, enquanto consumidores dependem de vários atributos opcionais não declarados no contrato.
   - Sintoma: repetição de `getattr(config, "...", default)` e defaults espalhados.
   - Impacto SOLID: viola ISP e enfraquece DIP (dependência em “detalhes implícitos”).

4. **Uso de `SkipValidation`/`object` para serviços do container**
   - O modelo de registro aceita `service: Annotated[object, SkipValidation]`.
   - Risco: desloca validação para runtime tardio; aumenta chance de erro em produção.

## Best practices recomendadas

### 1) Protocols mínimos por caso de uso (ISP/DIP)

- Criar contratos menores para recursos opcionais de configuração (ex.: `RetryConfig`, `RateLimitConfig`,
  `CircuitBreakerConfig`), em vez de depender de `getattr`.
- Cada componente depende apenas do protocolo necessário.

### 2) Preferir tipos públicos e estáveis

- Evitar imports privados (`_ProtocolMeta`, `_model_construction`).
- Se necessário combinar comportamento, encapsular atrás de uma camada local com testes de compatibilidade e fallback explícito.

### 3) Pydantic way: validação explícita nas bordas

- Trocar campos `object + SkipValidation` por modelos discriminados (`Union` com `discriminator`) quando houver conjunto
  conhecido de tipos.
- Quando realmente dinâmico, validar no momento do registro (`model_validator`) e persistir metadados tipados (ex.:
  tipo, capacidades, versão).

### 4) Evitar “protocol detection” heurístico

- Para runtime checks, usar `@runtime_checkable` + `isinstance` apenas em pontos de fronteira.
- No domínio interno, confiar no type checker estático e em testes de contrato.

### 5) Centralizar defaults de configuração

- Defaults devem morar no próprio modelo de settings (Pydantic) e não espalhados em `getattr` em cada consumidor.
- Consumidores recebem objetos já válidos e completos.

## Plano incremental sugerido

1. **Fase 1**: introduzir protocolos específicos de config e reduzir `getattr` em dispatcher.
2. **Fase 2**: mover defaults para `FlextSettings`/modelos de config.
3. **Fase 3**: substituir validação manual de protocol por checks explícitos em bordas.
4. **Fase 4**: reduzir dependência de APIs internas de typing/pydantic.

## Checklist prático (PR review)

- [ ] O protocolo novo é mínimo e orientado ao caso de uso?
- [ ] Existe `getattr` de campo de config que deveria estar no contrato?
- [ ] Há import privado de biblioteca externa?
- [ ] `SkipValidation` foi usado sem validação compensatória?
- [ ] Defaults estão no modelo Pydantic em vez de espalhados?
