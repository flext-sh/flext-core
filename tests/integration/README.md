# Testes de Integração FLEXT Core

## Como rodar os testes automatizados

Execute na raiz do projeto:

```bash
cd flext-core
python -m pytest tests/integration/ -v
```

- Todos os testes automatizados estão em **test_wildcard_exports.py**.
- Cobrem: imports, exports, exceptions, constants, utilities, fluxo end-to-end e estabilidade.

## Scripts auxiliares

Os scripts a seguir NÃO fazem parte da suíte de testes automatizada, mas podem ser usados para inspeção manual dos exports do flext-core:

- `test_imports.py` — Verifica duplicatas e mostra os exports principais.
- `analyze_exports.py` — Faz análise detalhada e categorização dos exports.

Execute-os manualmente na raiz do projeto, se desejar:

```bash
python test_imports.py
python analyze_exports.py
```

---

**Padrão de testes consolidado, limpo e documentado.**
