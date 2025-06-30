"""Entity to model mappers for persistence layer."""

from typing import Any

from flext_core.domain.entities import Pipeline, PipelineExecution, Plugin
from flext_core.domain.value_objects import (
    ExecutionId,
    ExecutionStatus,
    PipelineId,
    PipelineName,
    PipelineStep,
    PluginId,
)


def map_pipeline_model_data(data: dict[str, Any]) -> Pipeline:
    """Map database model data to Pipeline entity."""
    return Pipeline(
        id=PipelineId(data["id"]),
        name=PipelineName(data["name"]),
        description=data.get("description"),
        configuration=data.get("configuration", {}),
        steps=data.get("steps", []),
        is_active=data.get("is_active", True),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )


def map_pipeline_entity_to_model_data(pipeline: Pipeline) -> dict[str, Any]:
    """Map Pipeline entity to database model data."""
    return {
        "id": str(pipeline.id),
        "name": str(pipeline.name),
        "description": pipeline.description,
        "configuration": pipeline.configuration,
        "steps": pipeline.steps,
        "is_active": pipeline.is_active,
        "created_at": pipeline.created_at,
        "updated_at": pipeline.updated_at,
    }


def map_pipeline_execution_model_data(data: dict[str, Any]) -> PipelineExecution:
    """Map database model data to PipelineExecution entity."""
    return PipelineExecution(
        id=ExecutionId(data["id"]),
        pipeline_id=PipelineId(data["pipeline_id"]),
        status=ExecutionStatus(data["status"]),
        parameters=data.get("parameters", {}),
        result=data.get("result"),
        error_message=data.get("error_message"),
        started_at=data.get("started_at"),
        completed_at=data.get("completed_at"),
    )


def map_pipeline_execution_entity_to_model_data(execution: PipelineExecution) -> dict[str, Any]:
    """Map PipelineExecution entity to database model data."""
    return {
        "id": str(execution.id),
        "pipeline_id": str(execution.pipeline_id),
        "status": execution.status.value,
        "parameters": execution.parameters,
        "result": execution.result,
        "error_message": execution.error_message,
        "started_at": execution.started_at,
        "completed_at": execution.completed_at,
    }


def map_plugin_model_data(data: dict[str, Any]) -> Plugin:
    """Map database model data to Plugin entity."""
    return Plugin(
        id=PluginId(data["id"]),
        name=data["name"],
        version=data["version"],
        description=data.get("description"),
        configuration=data.get("configuration", {}),
        is_enabled=data.get("is_enabled", True),
        installed_at=data.get("installed_at"),
    )


def map_plugin_entity_to_model_data(plugin: Plugin) -> dict[str, Any]:
    """Map Plugin entity to database model data."""
    return {
        "id": str(plugin.id),
        "name": plugin.name,
        "version": plugin.version,
        "description": plugin.description,
        "configuration": plugin.configuration,
        "is_enabled": plugin.is_enabled,
        "installed_at": plugin.installed_at,
    }


def map_pipeline_step_entity_to_model_data(step: PipelineStep) -> dict[str, Any]:
    """Map PipelineStep value object to database model data."""
    return {
        "name": step.name,
        "type": step.type,
        "configuration": step.configuration,
        "order": step.order,
        "dependencies": step.dependencies,
    }


def map_relationship_data(data: dict[str, Any]) -> dict[str, Any]:
    """Map relationship data for complex entities."""
    # For now, return as-is, but this can be extended for complex relationships
    return data
