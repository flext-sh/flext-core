# Instalação - FLEXT Core

**Guia completo de instalação e configuração inicial**

## 🎯 Requisitos do Sistema

### Requisitos Obrigatórios

- **Python 3.13+** (biblioteca é exclusiva para Python 3.13)
- **pip** ou **Poetry** para gerenciamento de dependências
- **Git** para versionamento (desenvolvimento)

### Verificação do Python

```bash
# Verificar versão do Python
python --version
# Deve retornar: Python 3.13.x

# Verificar se pip está disponível
pip --version

# Verificar Poetry (opcional, mas recomendado)
poetry --version
```

## 📦 Métodos de Instalação

### 1. Instalação via Poetry (Recomendado)

**Poetry oferece melhor gerenciamento de dependências e ambientes virtuais.**

```bash
# Instalar FLEXT Core
poetry add flext-core

# Ou especificar versão específica
poetry add flext-core@^1.0.0

# Para desenvolvimento
poetry add --group dev flext-core
```

### 2. Instalação via pip

```bash
# Instalação básica
pip install flext-core

# Instalar versão específica
pip install flext-core==1.0.0

# Instalar versão mais recente
pip install --upgrade flext-core

# Instalar em ambiente virtual (recomendado)
python -m venv flext-env
source flext-env/bin/activate  # Linux/Mac
# flext-env\Scripts\activate   # Windows
pip install flext-core
```

### 3. Instalação para Desenvolvimento

```bash
# Clonar repositório
git clone https://github.com/flext/flext-core.git
cd flext-core

# Instalar Poetry (se não tiver)
curl -sSL https://install.python-poetry.org | python3 -

# Instalar dependências de desenvolvimento
poetry install

# Ativar ambiente virtual do Poetry
poetry shell

# Verificar instalação
make check
```

## 🔧 Configuração Inicial

### 1. Verificação da Instalação

```python
# test_installation.py
from flext_core import FlextResult, FlextContainer

def test_basic_functionality():
    """Teste básico da funcionalidade."""
    # Teste FlextResult
    result = FlextResult.ok("Installation successful!")
    assert result.is_success
    print(f"✅ FlextResult: {result.data}")
    
    # Teste FlextContainer
    container = FlextContainer()
    reg_result = container.register("test_service", "test_value")
    assert reg_result.is_success
    
    get_result = container.get("test_service")
    assert get_result.is_success
    assert get_result.data == "test_value"
    print("✅ FlextContainer: OK")
    
    print("🎉 FLEXT Core instalado e funcionando corretamente!")

if __name__ == "__main__":
    test_basic_functionality()
```

```bash
# Executar teste
python test_installation.py
```

### 2. Configuração do Ambiente

#### Variáveis de Ambiente (Opcional)

```bash
# .env (opcional)
FLEXT_DEBUG=false
FLEXT_LOG_LEVEL=INFO
FLEXT_ENVIRONMENT=production
FLEXT_MAX_CONNECTIONS=10
FLEXT_CACHE_TTL=3600
```

#### Carregamento da Configuração

```python
# config.py
from flext_core import FlextCoreSettings

# Carregamento automático das variáveis de ambiente
settings = FlextCoreSettings()

print(f"Debug: {settings.debug}")
print(f"Log Level: {settings.log_level}")
print(f"Environment: {settings.environment}")
```

### 3. Estrutura de Projeto Recomendada

```
meu_projeto/
├── src/
│   └── meu_projeto/
│       ├── __init__.py
│       ├── commands/          # Commands e handlers
│       ├── domain/            # Entidades e value objects
│       ├── infrastructure/    # Implementações técnicas
│       └── application/       # Serviços de aplicação
├── tests/
│   ├── __init__.py
│   ├── unit/                  # Testes unitários
│   └── integration/           # Testes de integração
├── docs/                      # Documentação
├── pyproject.toml            # Dependências Poetry
├── README.md
└── .env                      # Configurações locais
```

## 🏗️ Setup de Desenvolvimento

### 1. Ambiente de Desenvolvimento Completo

```bash
# Clonar e configurar projeto
git clone https://github.com/flext/flext-core.git
cd flext-core

# Instalar todas as dependências
make setup

# Ou manualmente:
poetry install --with dev,test,docs
poetry shell

# Instalar pre-commit hooks
make pre-commit
# Ou: pre-commit install
```

### 2. Comandos de Desenvolvimento

```bash
# Verificação completa (OBRIGATÓRIO antes de commit)
make validate              # lint + type-check + security + test

# Comandos individuais
make lint                  # Ruff linting PEP8 strict
make type-check            # MyPy strict mode
make test                  # Testes com coverage 90%+
make security              # Verificações de segurança

# Formatação de código
make format                # Auto-formatar código PEP8
make format-check          # Verificar formatação

# Desenvolvimento
make dev-install           # Setup ambiente desenvolvimento
make clean                 # Limpar arquivos temporários
```

### 3. Configuração do IDE

#### VS Code

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.rulers": [79],
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true
}
```

#### PyCharm

```python
# Configuração do interpretador:
# File > Settings > Project > Python Interpreter
# Selecionar: Poetry Environment (.venv/bin/python)

# Configurar Ruff como linter:
# File > Settings > Tools > External Tools
# Name: Ruff
# Program: ruff
# Arguments: check $FilePath$
```

## 🧪 Verificação da Instalação

### 1. Teste Completo de Funcionalidades

```python
# comprehensive_test.py
from flext_core import (
    FlextResult,
    FlextContainer,
    FlextCoreSettings,
    FlextCommand,
    FlextValidator,
    FlextValidationResult
)
from flext_core.patterns import (
    FlextCommandHandler,
    FlextMessageHandler,
    NotEmptyRule
)

def test_all_components():
    """Teste abrangente de todos os componentes."""
    
    print("🧪 Testando FLEXT Core...")
    
    # 1. FlextResult
    result = FlextResult.ok("Success")
    assert result.is_success
    print("✅ FlextResult: OK")
    
    # 2. FlextContainer
    container = FlextContainer()
    container.register("service", "value")
    get_result = container.get("service")
    assert get_result.is_success
    print("✅ FlextContainer: OK")
    
    # 3. FlextCoreSettings
    settings = FlextCoreSettings()
    assert hasattr(settings, 'debug')
    print("✅ FlextCoreSettings: OK")
    
    # 4. Command Pattern
    class TestCommand(FlextCommand):
        def validate(self) -> FlextResult[None]:
            return FlextResult.ok(None)
    
    class TestHandler(FlextCommandHandler[TestCommand, str]):
        def can_handle(self, command):
            return isinstance(command, TestCommand)
        
        def handle(self, command):
            return FlextResult.ok("handled")
    
    command = TestCommand()
    handler = TestHandler()
    cmd_result = handler.process_command(command)
    assert cmd_result.is_success
    print("✅ Command Pattern: OK")
    
    # 5. Validation
    class TestValidator(FlextValidator[str]):
        def validate_business_rules(self, data: str) -> FlextValidationResult:
            return FlextValidationResult.success()
    
    validator = TestValidator()
    validation_result = validator.validate("test")
    assert validation_result.is_valid
    print("✅ Validation: OK")
    
    print("🎉 Todos os componentes funcionando corretamente!")
    return True

if __name__ == "__main__":
    test_all_components()
```

### 2. Benchmark de Performance

```python
# performance_test.py
import time
from flext_core import FlextResult, FlextContainer

def benchmark_flext_result():
    """Benchmark FlextResult performance."""
    start = time.time()
    
    for i in range(10000):
        result = FlextResult.ok(f"value_{i}")
        if result.is_success:
            data = result.data
    
    end = time.time()
    print(f"⚡ FlextResult: 10k operations in {end - start:.4f}s")

def benchmark_container():
    """Benchmark Container performance."""
    container = FlextContainer()
    
    # Setup
    for i in range(1000):
        container.register(f"service_{i}", f"value_{i}")
    
    # Benchmark get operations
    start = time.time()
    
    for i in range(1000):
        result = container.get(f"service_{i}")
        if result.is_success:
            data = result.data
    
    end = time.time()
    print(f"⚡ Container: 1k lookups in {end - start:.4f}s")

if __name__ == "__main__":
    benchmark_flext_result()
    benchmark_container()
```

## 🔍 Troubleshooting

### Problemas Comuns

#### 1. Python Version Error

```bash
ERROR: FLEXT Core requires Python 3.13+
```

**Solução:**

```bash
# Instalar Python 3.13
# Ubuntu/Debian
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13 python3.13-venv

# macOS (Homebrew)
brew install python@3.13

# Windows
# Baixar de python.org
```

#### 2. Import Error

```python
ImportError: No module named 'flext_core'
```

**Soluções:**

```bash
# Verificar instalação
pip list | grep flext-core

# Reinstalar
pip uninstall flext-core
pip install flext-core

# Verificar ambiente virtual
which python
which pip
```

#### 3. Dependency Conflicts

```bash
ERROR: pip's dependency resolver does not currently consider all the packages
```

**Soluções:**

```bash
# Usar ambiente virtual limpo
python -m venv fresh-env
source fresh-env/bin/activate
pip install flext-core

# Ou usar Poetry
poetry init
poetry add flext-core
```

#### 4. Permission Errors (Windows)

```bash
ERROR: Could not install packages due to an EnvironmentError: [WinError 5]
```

**Soluções:**

```bash
# Instalar para usuário atual
pip install --user flext-core

# Ou executar como REDACTED_LDAP_BIND_PASSWORDistrador
# Ou usar ambiente virtual
```

### Verificação de Saúde do Sistema

```python
# health_check.py
import sys
import importlib.util

def check_system_health():
    """Verificação completa da saúde do sistema."""
    
    print("🔍 FLEXT Core Health Check")
    print("=" * 50)
    
    # Python version
    python_version = sys.version_info
    print(f"Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 13):
        print("❌ Python 3.13+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Check FLEXT Core installation
    try:
        import flext_core
        print(f"✅ FLEXT Core: v{flext_core.__version__}")
    except ImportError as e:
        print(f"❌ FLEXT Core not installed: {e}")
        return False
    
    # Check core modules
    modules = [
        'flext_core.result',
        'flext_core.container', 
        'flext_core.patterns.commands',
        'flext_core.patterns.handlers',
        'flext_core.patterns.validation'
    ]
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False
    
    print("=" * 50)
    print("🎉 Sistema saudável - FLEXT Core pronto para uso!")
    return True

if __name__ == "__main__":
    check_system_health()
```

## 📚 Próximos Passos

Após instalação bem-sucedida:

1. **[Quickstart](quickstart.md)** - Primeiros passos com FLEXT Core
2. **[Arquitetura](../architecture/overview.md)** - Entender a arquitetura
3. **[API Core](../api/core.md)** - Referência das APIs principais
4. **[Patterns](../api/patterns.md)** - Padrões avançados
5. **[Examples](../examples/overview.md)** - Exemplos práticos

## 🆘 Suporte

Se encontrou problemas na instalação:

- **Issues**: [GitHub Issues](https://github.com/flext/flext-core/issues)
- **Documentação**: [Docs Completa](https://docs.flext.dev)
- **Discussões**: [GitHub Discussions](https://github.com/flext/flext-core/discussions)

---

**FLEXT Core** está pronto para acelerar seu desenvolvimento empresarial!
