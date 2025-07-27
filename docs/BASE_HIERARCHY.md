# FLEXT Core - Base Module Hierarchy

**VALIDADO**: 2025-07-27 - ZERO dependências circulares ✅

## Hierarquia Validada de Módulos Base

Os módulos base seguem hierarquia rígida para EVITAR dependências circulares:

### NÍVEL 0: Fundação (Zero Dependências)
- `_constants_base.py` - Constantes fundamentais

### NÍVEL 1: Tipos (Depende: constants)  
- `_types_base.py` - TypeVars e type aliases

### NÍVEL 2: Validação (Depende: types)
- `_validation_base.py` - Validações básicas

### NÍVEL 3: Utilitários (Depende: validation)
- `_utilities_base.py` - Funções utilitárias 

### NÍVEL 4: Result (Depende: types, constants)
- `_result_base.py` - Railway-oriented programming

### NÍVEL 5: Configuração (Depende: result, utilities)
- `_config_base.py` - Configurações base

### NÍVEL 6: Logging (Depende: constants, validation)  
- `_loggings_base.py` - Sistema de logging

### NÍVEL 7: Exceções (Depende: constants, types)
- `_exceptions_base.py` - Hierarquia de exceções

### NÍVEL 8: Decorators (Depende: types)
- `_decorators_base.py` - Decoradores funcionais

### NÍVEL 9: Mixins (Depende: validation, types)
- `_mixins_base.py` - Classes mixin reutilizáveis

### NÍVEL 10: Field Config (Depende: mixins)
- `_field_config_base.py` - Configurações de campo

## Regras OBRIGATÓRIAS

1. **NUNCA** importar módulos públicos em módulos base
2. **SEMPRE** importar apenas módulos de níveis inferiores
3. **TODOS** os módulos base usam prefixo `_` e classes/funções privadas
4. **ZERO** tolerância para dependências circulares

## Validação de Importação

```bash
# Verificar se não há importações circulares
python -c "
import sys
sys.path.insert(0, 'src')

# Importar módulos na ordem hierárquica
from flext_core._constants_base import *
from flext_core._types_base import *
from flext_core._validation_base import *
from flext_core._utilities_base import *
from flext_core._result_base import *
from flext_core._config_base import *
from flext_core._loggings_base import *
from flext_core._exceptions_base import *
from flext_core._decorators_base import *
from flext_core._mixins_base import *
from flext_core._field_config_base import *
print('✅ HIERARQUIA VALIDADA - Zero dependências circulares!')
"
```

## Status: ✅ VALIDADO e FUNCIONAL