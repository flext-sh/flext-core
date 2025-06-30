"""Domain commands for FLEXT Meltano Enterprise."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from flext_core.domain.value_objects import PipelineId, PipelineStep

# Import the canonical CreatePipelineCommand from the unified commands module


class AddStepToPipelineCommand(BaseModel):
    """AddStepToPipelineCommand - Command Object.

    Implementa um objeto de comando seguindo padrões CQRS com validação Pydantic integrada. Encapsula uma operação de negócio específica com parâmetros validados.

    Arquitetura: Command Pattern + CQRS
    Validação: Pydantic v2 com Python 3.13 type hints
    Padrões: Immutable objects, validation by construction

    Attributes: pipeline_id (PipelineId): Atributo da classe.
    step (PipelineStep): Atributo da classe.

    Methods: Sem métodos públicos.

    Examples: Uso típico da classe:

    ```python
    addsteptopipelinecommand = AddStepToPipelineCommand(param='value')
    result = await handler.execute(addsteptopipelinecommand)
    ```

    See Also
    --------
    - [Documentação da Arquitetura](../../docs/architecture/index.md)
    - [Padrões de Design](../../docs/architecture/001-clean-architecture-ddd.md)

    Note: Esta classe segue os padrões CQRS e Command Pattern estabelecidos no projeto.

    """

    """Command to add a step to an existing pipeline."""

    pipeline_id: PipelineId
    step: PipelineStep


class UpdatePipelineScheduleCommand(BaseModel):
    """UpdatePipelineScheduleCommand - Command Object.

    Implementa um objeto de comando seguindo padrões CQRS com validação Pydantic integrada. Encapsula uma operação de negócio específica com parâmetros validados.

    Arquitetura: Command Pattern + CQRS
    Validação: Pydantic v2 com Python 3.13 type hints
    Padrões: Immutable objects, validation by construction

    Attributes: pipeline_id (PipelineId): Atributo da classe.
    schedule (str | None): Atributo da classe.

    Methods: Sem métodos públicos.

    Examples: Uso típico da classe:

    ```python
    updatepipelineschedulecommand = UpdatePipelineScheduleCommand(param='value')
    result = await handler.execute(updatepipelineschedulecommand)
    ```

    See Also
    --------
    - [Documentação da Arquitetura](../../docs/architecture/index.md)
    - [Padrões de Design](../../docs/architecture/001-clean-architecture-ddd.md)

    Note: Esta classe segue os padrões CQRS e Command Pattern estabelecidos no projeto.

    """

    """Command to update the schedule of a pipeline."""

    pipeline_id: PipelineId
    schedule: str | None


class RunPipelineCommand(BaseModel):
    """RunPipelineCommand - Command Object.

    Implementa um objeto de comando seguindo padrões CQRS com validação Pydantic integrada. Encapsula uma operação de negócio específica com parâmetros validados.

    Arquitetura: Command Pattern + CQRS
    Validação: Pydantic v2 com Python 3.13 type hints
    Padrões: Immutable objects, validation by construction

    Attributes: pipeline_id (PipelineId): Atributo da classe.
    triggered_by (str): Atributo da classe.

    Methods: Sem métodos públicos.

    Examples: Uso típico da classe:

    ```python
    runpipelinecommand = RunPipelineCommand(param='value')
    result = await handler.execute(runpipelinecommand)
    ```

    See Also
    --------
    - [Documentação da Arquitetura](../../docs/architecture/index.md)
    - [Padrões de Design](../../docs/architecture/001-clean-architecture-ddd.md)

    Note: Esta classe segue os padrões CQRS e Command Pattern estabelecidos no projeto.

    """

    """Command to trigger a pipeline run."""

    pipeline_id: PipelineId
    triggered_by: str


# Import the canonical UpdatePipelineCommand from the unified commands module


# Import the canonical ExecutePipelineCommand from the unified commands module


# Import the canonical DeletePipelineCommand from the unified commands module


# Import the canonical commands from unified modules


# Import the canonical RunFullE2ECommand from unified module


# ZERO TOLERANCE: Import E2E commands from canonical location to eliminate duplicates
