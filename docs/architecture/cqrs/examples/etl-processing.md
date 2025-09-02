# ETL Processing CQRS Examples

**Version**: 1.0  
**Target**: Data Engineers, ETL Developers  
**Framework**: Meltano, Apache Airflow, Custom ETL  
**Complexity**: Advanced

## 📋 Overview

This document provides practical examples of implementing FlextCommands CQRS patterns in ETL (Extract, Transform, Load) processing workflows. It covers data pipeline operations, transformation commands, data quality validation, and integration with popular Python data processing frameworks.

## 🎯 Key Benefits

- ✅ **Structured Operations**: ETL operations as validated commands
- ✅ **Data Lineage**: Automatic tracking of data transformations
- ✅ **Error Recovery**: Structured error handling and retry logic
- ✅ **Quality Gates**: Data validation at each processing stage
- ✅ **Monitoring**: Real-time visibility into ETL operations

---

## 🚀 Data Extraction Commands

### Extract from Database Source

```python
from typing import Dict, List, object, Optional
from datetime import datetime, timedelta
from flext_core import FlextCommands, FlextResult
from enum import Enum

class ExtractMode(str, Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DELTA = "delta"
    SNAPSHOT = "snapshot"

class ExtractDataCommand(FlextCommands.Models.Command):
    """Command to extract data from a source system."""
    
    source_name: str
    table_name: str
    extract_mode: ExtractMode = ExtractMode.INCREMENTAL
    
    # Time-based filters
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Incremental tracking
    last_updated_column: Optional[str] = "updated_at"
    watermark_value: Optional[str] = None
    
    # Filtering and partitioning
    where_clause: Optional[str] = None
    partition_column: Optional[str] = None
    partition_values: List[str] = Field(default_factory=list)
    
    # Output configuration
    output_format: str = "parquet"  # parquet, json, csv
    compression: Optional[str] = "snappy"
    batch_size: int = 10000
    
    # Quality controls
    max_records: Optional[int] = None
    validate_schema: bool = True
    
    def validate_command(self) -> FlextResult[None]:
        """Validate extraction command parameters."""
        
        # Validate source and table
        validation_result = (
            self.require_field("source_name", self.source_name)
            .flat_map(lambda _: self.require_field("table_name", self.table_name))
        )
        
        if validation_result.is_failure:
            return validation_result
        
        # Validate time range for incremental extracts
        if self.extract_mode == ExtractMode.INCREMENTAL:
            if not self.last_updated_column:
                return FlextResult[None].fail(
                    "last_updated_column required for incremental extracts"
                )
        
        # Validate date range
        if self.start_date and self.end_date:
            if self.start_date >= self.end_date:
                return FlextResult[None].fail(
                    "start_date must be before end_date"
                )
        
        # Validate output format
        valid_formats = {"parquet", "json", "csv", "avro"}
        if self.output_format not in valid_formats:
            return FlextResult[None].fail(
                f"Invalid output_format. Valid options: {valid_formats}"
            )
        
        # Validate batch size
        if self.batch_size < 1 or self.batch_size > 100000:
            return FlextResult[None].fail(
                "batch_size must be between 1 and 100,000"
            )
        
        return FlextResult[None].ok(None)

class ExtractDataHandler(FlextCommands.Handlers.CommandHandler[ExtractDataCommand, Dict[str, object]]):
    """Handler for data extraction operations."""
    
    def __init__(self, 
                 source_service: DataSourceService,
                 metadata_service: MetadataService,
                 storage_service: StorageService,
                 quality_service: DataQualityService):
        super().__init__(handler_name="ExtractDataHandler")
        self.source_service = source_service
        self.metadata_service = metadata_service
        self.storage_service = storage_service
        self.quality_service = quality_service
    
    def handle(self, command: ExtractDataCommand) -> FlextResult[Dict[str, object]]:
        """Execute data extraction with comprehensive error handling."""
        
        extraction_start = datetime.utcnow()
        extraction_id = f"extract_{command.source_name}_{command.table_name}_{int(extraction_start.timestamp())}"
        
        try:
            self.log_info("Starting data extraction",
                         extraction_id=extraction_id,
                         source=command.source_name,
                         table=command.table_name,
                         mode=command.extract_mode)
            
            # Get source connection
            source_connection = self.source_service.get_connection(command.source_name)
            if not source_connection:
                return FlextResult[Dict[str, object]].fail(
                    f"Source connection '{command.source_name}' not found",
                    error_code="SOURCE_NOT_FOUND"
                )
            
            # Validate source table exists
            if not source_connection.table_exists(command.table_name):
                return FlextResult[Dict[str, object]].fail(
                    f"Table '{command.table_name}' not found in source '{command.source_name}'",
                    error_code="TABLE_NOT_FOUND"
                )
            
            # Get table schema for validation
            if command.validate_schema:
                source_schema = source_connection.get_table_schema(command.table_name)
                stored_schema = self.metadata_service.get_schema(command.source_name, command.table_name)
                
                if stored_schema and not self._schemas_compatible(source_schema, stored_schema):
                    return FlextResult[Dict[str, object]].fail(
                        f"Schema mismatch detected for table '{command.table_name}'",
                        error_code="SCHEMA_MISMATCH"
                    )
            
            # Build extraction query
            query_result = self._build_extraction_query(command, source_connection)
            if query_result.is_failure:
                return FlextResult[Dict[str, object]].fail(f"Query building failed: {query_result.error}")
            
            extraction_query = query_result.value
            
            # Execute extraction in batches
            total_records = 0
            batch_count = 0
            output_files = []
            
            self.log_info("Executing extraction query", 
                         extraction_id=extraction_id,
                         query=extraction_query[:200] + "..." if len(extraction_query) > 200 else extraction_query)
            
            for batch_data in source_connection.execute_query_batches(extraction_query, command.batch_size):
                batch_count += 1
                batch_records = len(batch_data)
                total_records += batch_records
                
                # Check max records limit
                if command.max_records and total_records > command.max_records:
                    self.log_warning("Max records limit reached",
                                   extraction_id=extraction_id,
                                   limit=command.max_records,
                                   current_total=total_records)
                    break
                
                # Apply data quality checks
                quality_result = self.quality_service.validate_batch(
                    data=batch_data,
                    source=command.source_name,
                    table=command.table_name
                )
                
                if quality_result.is_failure:
                    return FlextResult[Dict[str, object]].fail(
                        f"Data quality validation failed: {quality_result.error}",
                        error_code="QUALITY_CHECK_FAILED"
                    )
                
                # Store batch
                output_path = f"{command.source_name}/{command.table_name}/batch_{batch_count:05d}.{command.output_format}"
                storage_result = self.storage_service.store_data(
                    data=batch_data,
                    path=output_path,
                    format=command.output_format,
                    compression=command.compression
                )
                
                if storage_result.is_failure:
                    return FlextResult[Dict[str, object]].fail(
                        f"Data storage failed: {storage_result.error}",
                        error_code="STORAGE_FAILED"
                    )
                
                output_files.append(storage_result.value["file_path"])
                
                self.log_info("Batch processed",
                             extraction_id=extraction_id,
                             batch=batch_count,
                             records=batch_records,
                             total_records=total_records)
            
            extraction_end = datetime.utcnow()
            duration = (extraction_end - extraction_start).total_seconds()
            
            # Update watermark for incremental extracts
            new_watermark = None
            if command.extract_mode == ExtractMode.INCREMENTAL and total_records > 0:
                new_watermark = self._calculate_new_watermark(command, source_connection)
                if new_watermark:
                    self.metadata_service.update_watermark(
                        source=command.source_name,
                        table=command.table_name,
                        column=command.last_updated_column,
                        value=new_watermark
                    )
            
            # Record extraction metadata
            extraction_metadata = {
                "extraction_id": extraction_id,
                "source_name": command.source_name,
                "table_name": command.table_name,
                "extract_mode": command.extract_mode,
                "started_at": extraction_start.isoformat(),
                "completed_at": extraction_end.isoformat(),
                "duration_seconds": duration,
                "total_records": total_records,
                "batch_count": batch_count,
                "output_files": output_files,
                "output_format": command.output_format,
                "compression": command.compression,
                "new_watermark": new_watermark,
                "data_size_mb": sum(self.storage_service.get_file_size(f) for f in output_files) / (1024 * 1024)
            }
            
            self.metadata_service.record_extraction(extraction_metadata)
            
            self.log_info("Data extraction completed successfully",
                         extraction_id=extraction_id,
                         total_records=total_records,
                         duration_seconds=duration,
                         throughput_records_per_sec=total_records / duration if duration > 0 else 0)
            
            return FlextResult[Dict[str, object]].ok(extraction_metadata)
            
        except Exception as e:
            self.log_error("Data extraction failed",
                          extraction_id=extraction_id,
                          error=str(e))
            return FlextResult[Dict[str, object]].fail(f"Extraction failed: {e}")
    
    def _build_extraction_query(self, command: ExtractDataCommand, connection) -> FlextResult[str]:
        """Build SQL query based on extraction parameters."""
        try:
            base_query = f"SELECT * FROM {command.table_name}"
            where_conditions = []
            
            # Add incremental conditions
            if command.extract_mode == ExtractMode.INCREMENTAL and command.watermark_value:
                where_conditions.append(f"{command.last_updated_column} > '{command.watermark_value}'")
            
            # Add date range filters
            if command.start_date:
                date_column = command.last_updated_column or "created_at"
                where_conditions.append(f"{date_column} >= '{command.start_date.isoformat()}'")
            
            if command.end_date:
                date_column = command.last_updated_column or "created_at"
                where_conditions.append(f"{date_column} < '{command.end_date.isoformat()}'")
            
            # Add custom where clause
            if command.where_clause:
                where_conditions.append(f"({command.where_clause})")
            
            # Add partition filters
            if command.partition_column and command.partition_values:
                partition_filter = f"{command.partition_column} IN ({','.join(f\"'{v}'\" for v in command.partition_values)})"
                where_conditions.append(partition_filter)
            
            # Combine conditions
            if where_conditions:
                base_query += " WHERE " + " AND ".join(where_conditions)
            
            # Add ordering for consistent results
            if command.extract_mode == ExtractMode.INCREMENTAL and command.last_updated_column:
                base_query += f" ORDER BY {command.last_updated_column}"
            
            return FlextResult[str].ok(base_query)
            
        except Exception as e:
            return FlextResult[str].fail(f"Query building failed: {e}")
    
    def _calculate_new_watermark(self, command: ExtractDataCommand, connection) -> Optional[str]:
        """Calculate new watermark value for incremental extracts."""
        try:
            query = f"SELECT MAX({command.last_updated_column}) as max_value FROM {command.table_name}"
            result = connection.execute_query(query)
            if result and len(result) > 0 and result[0]['max_value']:
                return str(result[0]['max_value'])
        except Exception as e:
            self.log_warning("Failed to calculate new watermark", error=str(e))
        return None
    
    def _schemas_compatible(self, source_schema: Dict, stored_schema: Dict) -> bool:
        """Check if schemas are compatible."""
        # Simplified schema comparison
        source_columns = set(source_schema.get('columns', {}).keys())
        stored_columns = set(stored_schema.get('columns', {}).keys())
        
        # Allow new columns in source (schema evolution)
        # But don't allow removing columns
        missing_columns = stored_columns - source_columns
        return len(missing_columns) == 0
```

---

## 🔄 Data Transformation Commands

### Transform Data Command

```python
from typing import Callable, Union
import pandas as pd

class TransformationRule(BaseModel):
    """Definition of a data transformation rule."""
    
    rule_name: str
    rule_type: str  # "column_rename", "data_type", "calculation", "filter", "aggregation"
    source_columns: List[str]
    target_column: str
    parameters: Dict[str, object] = Field(default_factory=dict)
    condition: Optional[str] = None  # SQL-like condition

class TransformDataCommand(FlextCommands.Models.Command):
    """Command to transform extracted data."""
    
    input_source: str  # Path or reference to input data
    transformation_rules: List[TransformationRule]
    output_destination: str
    
    # Processing options
    chunk_size: int = 10000
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Quality controls
    validate_output: bool = True
    expected_output_schema: Optional[Dict[str, str]] = None
    data_quality_rules: List[str] = Field(default_factory=list)
    
    # Error handling
    error_handling: str = "fail"  # "fail", "skip", "log"
    max_error_percentage: float = 5.0
    
    def validate_command(self) -> FlextResult[None]:
        """Validate transformation command."""
        
        # Basic field validation
        validation_result = (
            self.require_field("input_source", self.input_source)
            .flat_map(lambda _: self.require_field("output_destination", self.output_destination))
        )
        
        if validation_result.is_failure:
            return validation_result
        
        # Validate transformation rules
        if not self.transformation_rules:
            return FlextResult[None].fail("At least one transformation rule is required")
        
        valid_rule_types = {
            "column_rename", "data_type", "calculation", "filter", 
            "aggregation", "join", "pivot", "unpivot"
        }
        
        for i, rule in enumerate(self.transformation_rules):
            if rule.rule_type not in valid_rule_types:
                return FlextResult[None].fail(
                    f"Invalid rule type '{rule.rule_type}' in rule {i}. "
                    f"Valid types: {valid_rule_types}"
                )
            
            if not rule.source_columns:
                return FlextResult[None].fail(f"source_columns required for rule {i}")
        
        # Validate processing parameters
        if self.chunk_size < 1 or self.chunk_size > 1000000:
            return FlextResult[None].fail("chunk_size must be between 1 and 1,000,000")
        
        if self.max_workers < 1 or self.max_workers > 16:
            return FlextResult[None].fail("max_workers must be between 1 and 16")
        
        # Validate error handling
        valid_error_handling = {"fail", "skip", "log"}
        if self.error_handling not in valid_error_handling:
            return FlextResult[None].fail(f"Invalid error_handling. Valid options: {valid_error_handling}")
        
        if not 0 <= self.max_error_percentage <= 100:
            return FlextResult[None].fail("max_error_percentage must be between 0 and 100")
        
        return FlextResult[None].ok(None)

class TransformDataHandler(FlextCommands.Handlers.CommandHandler[TransformDataCommand, Dict[str, object]]):
    """Handler for data transformation operations."""
    
    def __init__(self, 
                 storage_service: StorageService,
                 transformation_engine: TransformationEngine,
                 quality_service: DataQualityService,
                 metadata_service: MetadataService):
        super().__init__(handler_name="TransformDataHandler")
        self.storage_service = storage_service
        self.transformation_engine = transformation_engine
        self.quality_service = quality_service
        self.metadata_service = metadata_service
    
    def handle(self, command: TransformDataCommand) -> FlextResult[Dict[str, object]]:
        """Execute data transformation with error handling."""
        
        transformation_start = datetime.utcnow()
        transformation_id = f"transform_{int(transformation_start.timestamp())}"
        
        try:
            self.log_info("Starting data transformation",
                         transformation_id=transformation_id,
                         input_source=command.input_source,
                         rules_count=len(command.transformation_rules))
            
            # Load input data
            input_data_result = self.storage_service.load_data(command.input_source)
            if input_data_result.is_failure:
                return FlextResult[Dict[str, object]].fail(
                    f"Failed to load input data: {input_data_result.error}",
                    error_code="INPUT_LOAD_FAILED"
                )
            
            input_data = input_data_result.value
            total_input_records = len(input_data)
            
            self.log_info("Input data loaded",
                         transformation_id=transformation_id,
                         records_count=total_input_records)
            
            # Process data in chunks
            transformed_chunks = []
            error_count = 0
            processed_records = 0
            
            for chunk_start in range(0, total_input_records, command.chunk_size):
                chunk_end = min(chunk_start + command.chunk_size, total_input_records)
                chunk_data = input_data[chunk_start:chunk_end]
                
                # Apply transformations to chunk
                chunk_result = self._transform_chunk(
                    chunk_data, 
                    command.transformation_rules,
                    transformation_id,
                    chunk_start // command.chunk_size + 1
                )
                
                if chunk_result.is_failure:
                    error_count += len(chunk_data)
                    error_percentage = (error_count / total_input_records) * 100
                    
                    if command.error_handling == "fail":
                        return FlextResult[Dict[str, object]].fail(
                            f"Transformation failed: {chunk_result.error}",
                            error_code="TRANSFORMATION_FAILED"
                        )
                    elif error_percentage > command.max_error_percentage:
                        return FlextResult[Dict[str, object]].fail(
                            f"Error rate {error_percentage:.2f}% exceeds maximum {command.max_error_percentage}%",
                            error_code="ERROR_RATE_EXCEEDED"
                        )
                    else:
                        self.log_warning("Chunk transformation failed, continuing",
                                       transformation_id=transformation_id,
                                       chunk_start=chunk_start,
                                       error=chunk_result.error)
                        continue
                
                transformed_chunks.append(chunk_result.value)
                processed_records += len(chunk_data)
                
                self.log_info("Chunk transformed",
                             transformation_id=transformation_id,
                             chunk_records=len(chunk_data),
                             processed_records=processed_records,
                             total_records=total_input_records)
            
            # Combine transformed chunks
            if not transformed_chunks:
                return FlextResult[Dict[str, object]].fail(
                    "No data successfully transformed",
                    error_code="NO_DATA_TRANSFORMED"
                )
            
            transformed_data = pd.concat(transformed_chunks, ignore_index=True)
            
            # Validate output schema if specified
            if command.expected_output_schema:
                schema_validation = self._validate_output_schema(
                    transformed_data, 
                    command.expected_output_schema
                )
                if schema_validation.is_failure:
                    return FlextResult[Dict[str, object]].fail(
                        f"Output schema validation failed: {schema_validation.error}",
                        error_code="SCHEMA_VALIDATION_FAILED"
                    )
            
            # Apply data quality rules
            if command.validate_output and command.data_quality_rules:
                quality_result = self.quality_service.validate_data(
                    data=transformed_data,
                    rules=command.data_quality_rules
                )
                if quality_result.is_failure:
                    return FlextResult[Dict[str, object]].fail(
                        f"Data quality validation failed: {quality_result.error}",
                        error_code="QUALITY_VALIDATION_FAILED"
                    )
            
            # Save transformed data
            output_result = self.storage_service.save_data(
                data=transformed_data,
                destination=command.output_destination
            )
            
            if output_result.is_failure:
                return FlextResult[Dict[str, object]].fail(
                    f"Failed to save transformed data: {output_result.error}",
                    error_code="OUTPUT_SAVE_FAILED"
                )
            
            transformation_end = datetime.utcnow()
            duration = (transformation_end - transformation_start).total_seconds()
            
            # Record transformation metadata
            transformation_metadata = {
                "transformation_id": transformation_id,
                "input_source": command.input_source,
                "output_destination": command.output_destination,
                "started_at": transformation_start.isoformat(),
                "completed_at": transformation_end.isoformat(),
                "duration_seconds": duration,
                "input_records": total_input_records,
                "output_records": len(transformed_data),
                "error_records": error_count,
                "transformation_rules": len(command.transformation_rules),
                "throughput_records_per_sec": processed_records / duration if duration > 0 else 0
            }
            
            self.metadata_service.record_transformation(transformation_metadata)
            
            self.log_info("Data transformation completed successfully",
                         transformation_id=transformation_id,
                         input_records=total_input_records,
                         output_records=len(transformed_data),
                         duration_seconds=duration)
            
            return FlextResult[Dict[str, object]].ok(transformation_metadata)
            
        except Exception as e:
            self.log_error("Data transformation failed",
                          transformation_id=transformation_id,
                          error=str(e))
            return FlextResult[Dict[str, object]].fail(f"Transformation failed: {e}")
    
    def _transform_chunk(self, 
                        chunk_data: pd.DataFrame, 
                        rules: List[TransformationRule],
                        transformation_id: str,
                        chunk_number: int) -> FlextResult[pd.DataFrame]:
        """Apply transformation rules to a data chunk."""
        try:
            current_data = chunk_data.copy()
            
            for rule in rules:
                rule_result = self._apply_transformation_rule(current_data, rule)
                if rule_result.is_failure:
                    return FlextResult[pd.DataFrame].fail(
                        f"Rule '{rule.rule_name}' failed: {rule_result.error}"
                    )
                current_data = rule_result.value
            
            return FlextResult[pd.DataFrame].ok(current_data)
            
        except Exception as e:
            return FlextResult[pd.DataFrame].fail(f"Chunk transformation failed: {e}")
    
    def _apply_transformation_rule(self, 
                                  data: pd.DataFrame, 
                                  rule: TransformationRule) -> FlextResult[pd.DataFrame]:
        """Apply a single transformation rule."""
        try:
            if rule.rule_type == "column_rename":
                # Rename columns
                rename_map = dict(zip(rule.source_columns, [rule.target_column]))
                return FlextResult[pd.DataFrame].ok(data.rename(columns=rename_map))
            
            elif rule.rule_type == "data_type":
                # Convert data types
                target_type = rule.parameters.get("target_type")
                if not target_type:
                    return FlextResult[pd.DataFrame].fail("target_type parameter required")
                
                for col in rule.source_columns:
                    if col in data.columns:
                        data[col] = data[col].astype(target_type)
                return FlextResult[pd.DataFrame].ok(data)
            
            elif rule.rule_type == "calculation":
                # Apply calculation
                expression = rule.parameters.get("expression")
                if not expression:
                    return FlextResult[pd.DataFrame].fail("expression parameter required")
                
                # Simple expression evaluation (extend as needed)
                data[rule.target_column] = data.eval(expression)
                return FlextResult[pd.DataFrame].ok(data)
            
            elif rule.rule_type == "filter":
                # Filter data
                condition = rule.condition or rule.parameters.get("condition")
                if not condition:
                    return FlextResult[pd.DataFrame].fail("condition required for filter rule")
                
                filtered_data = data.query(condition)
                return FlextResult[pd.DataFrame].ok(filtered_data)
            
            elif rule.rule_type == "aggregation":
                # Group and aggregate
                group_by = rule.parameters.get("group_by", [])
                agg_func = rule.parameters.get("function", "sum")
                
                if group_by:
                    grouped_data = data.groupby(group_by)[rule.source_columns].agg(agg_func).reset_index()
                else:
                    # Global aggregation
                    agg_result = data[rule.source_columns].agg(agg_func)
                    grouped_data = pd.DataFrame([agg_result])
                
                return FlextResult[pd.DataFrame].ok(grouped_data)
            
            else:
                return FlextResult[pd.DataFrame].fail(f"Unsupported rule type: {rule.rule_type}")
                
        except Exception as e:
            return FlextResult[pd.DataFrame].fail(f"Rule application failed: {e}")
    
    def _validate_output_schema(self, 
                               data: pd.DataFrame, 
                               expected_schema: Dict[str, str]) -> FlextResult[None]:
        """Validate that output data matches expected schema."""
        try:
            # Check columns exist
            missing_columns = set(expected_schema.keys()) - set(data.columns)
            if missing_columns:
                return FlextResult[None].fail(f"Missing expected columns: {missing_columns}")
            
            # Check data types (simplified)
            for column, expected_type in expected_schema.items():
                actual_type = str(data[column].dtype)
                if expected_type not in actual_type:
                    return FlextResult[None].fail(
                        f"Column '{column}' has type '{actual_type}', expected '{expected_type}'"
                    )
            
            return FlextResult[None].ok(None)
            
        except Exception as e:
            return FlextResult[None].fail(f"Schema validation failed: {e}")
```

---

## 📊 Data Loading Commands

### Load Data Command

```python
class LoadDataCommand(FlextCommands.Models.Command):
    """Command to load transformed data into target system."""
    
    source_data: str  # Path or reference to transformed data
    target_connection: str
    target_table: str
    
    # Loading strategy
    load_mode: str = "append"  # "append", "overwrite", "upsert", "merge"
    merge_keys: List[str] = Field(default_factory=list)  # For upsert/merge operations
    
    # Performance settings
    batch_size: int = 5000
    parallel_loads: bool = False
    max_parallel_workers: int = 2
    
    # Data handling
    create_table_if_not_exists: bool = True
    truncate_before_load: bool = False
    deduplicate_data: bool = False
    deduplication_keys: List[str] = Field(default_factory=list)
    
    # Quality gates
    validate_before_load: bool = True
    post_load_validation: bool = True
    expected_record_count: Optional[int] = None
    
    def validate_command(self) -> FlextResult[None]:
        """Validate data loading parameters."""
        
        validation_result = (
            self.require_field("source_data", self.source_data)
            .flat_map(lambda _: self.require_field("target_connection", self.target_connection))
            .flat_map(lambda _: self.require_field("target_table", self.target_table))
        )
        
        if validation_result.is_failure:
            return validation_result
        
        # Validate load mode
        valid_modes = {"append", "overwrite", "upsert", "merge"}
        if self.load_mode not in valid_modes:
            return FlextResult[None].fail(f"Invalid load_mode. Valid options: {valid_modes}")
        
        # Validate merge keys for upsert/merge
        if self.load_mode in {"upsert", "merge"} and not self.merge_keys:
            return FlextResult[None].fail("merge_keys required for upsert/merge operations")
        
        # Validate deduplication settings
        if self.deduplicate_data and not self.deduplication_keys:
            return FlextResult[None].fail("deduplication_keys required when deduplicate_data is True")
        
        return FlextResult[None].ok(None)

class LoadDataHandler(FlextCommands.Handlers.CommandHandler[LoadDataCommand, Dict[str, object]]):
    """Handler for data loading operations."""
    
    def __init__(self,
                 storage_service: StorageService,
                 target_service: TargetSystemService,
                 quality_service: DataQualityService,
                 metadata_service: MetadataService):
        super().__init__(handler_name="LoadDataHandler")
        self.storage_service = storage_service
        self.target_service = target_service
        self.quality_service = quality_service
        self.metadata_service = metadata_service
    
    def handle(self, command: LoadDataCommand) -> FlextResult[Dict[str, object]]:
        """Execute data loading with comprehensive validation."""
        
        load_start = datetime.utcnow()
        load_id = f"load_{command.target_table}_{int(load_start.timestamp())}"
        
        try:
            self.log_info("Starting data load",
                         load_id=load_id,
                         source=command.source_data,
                         target_table=command.target_table,
                         load_mode=command.load_mode)
            
            # Load source data
            source_data_result = self.storage_service.load_data(command.source_data)
            if source_data_result.is_failure:
                return FlextResult[Dict[str, object]].fail(
                    f"Failed to load source data: {source_data_result.error}",
                    error_code="SOURCE_LOAD_FAILED"
                )
            
            source_data = source_data_result.value
            total_records = len(source_data)
            
            # Validate expected record count
            if command.expected_record_count and total_records != command.expected_record_count:
                return FlextResult[Dict[str, object]].fail(
                    f"Record count mismatch. Expected: {command.expected_record_count}, Got: {total_records}",
                    error_code="RECORD_COUNT_MISMATCH"
                )
            
            # Pre-load validation
            if command.validate_before_load:
                validation_result = self.quality_service.validate_for_loading(
                    data=source_data,
                    target_connection=command.target_connection,
                    target_table=command.target_table
                )
                if validation_result.is_failure:
                    return FlextResult[Dict[str, object]].fail(
                        f"Pre-load validation failed: {validation_result.error}",
                        error_code="PRE_LOAD_VALIDATION_FAILED"
                    )
            
            # Get target connection
            target_connection = self.target_service.get_connection(command.target_connection)
            if not target_connection:
                return FlextResult[Dict[str, object]].fail(
                    f"Target connection '{command.target_connection}' not found",
                    error_code="TARGET_CONNECTION_NOT_FOUND"
                )
            
            # Deduplicate data if requested
            if command.deduplicate_data:
                source_data = source_data.drop_duplicates(subset=command.deduplication_keys, keep='last')
                deduplicated_count = len(source_data)
                self.log_info("Data deduplicated",
                             load_id=load_id,
                             original_count=total_records,
                             deduplicated_count=deduplicated_count,
                             removed_count=total_records - deduplicated_count)
                total_records = deduplicated_count
            
            # Create target table if needed
            if command.create_table_if_not_exists and not target_connection.table_exists(command.target_table):
                create_result = target_connection.create_table_from_data(command.target_table, source_data)
                if not create_result:
                    return FlextResult[Dict[str, object]].fail(
                        f"Failed to create target table '{command.target_table}'",
                        error_code="TABLE_CREATION_FAILED"
                    )
                self.log_info("Target table created", load_id=load_id, table=command.target_table)
            
            # Truncate if requested
            if command.truncate_before_load:
                target_connection.truncate_table(command.target_table)
                self.log_info("Target table truncated", load_id=load_id, table=command.target_table)
            
            # Execute load based on mode
            load_result = self._execute_load_by_mode(
                target_connection, 
                source_data, 
                command,
                load_id
            )
            
            if load_result.is_failure:
                return FlextResult[Dict[str, object]].fail(
                    f"Data load failed: {load_result.error}",
                    error_code="LOAD_EXECUTION_FAILED"
                )
            
            load_stats = load_result.value
            
            # Post-load validation
            if command.post_load_validation:
                post_validation = self._validate_post_load(
                    target_connection,
                    command.target_table,
                    total_records,
                    command.load_mode
                )
                if post_validation.is_failure:
                    return FlextResult[Dict[str, object]].fail(
                        f"Post-load validation failed: {post_validation.error}",
                        error_code="POST_LOAD_VALIDATION_FAILED"
                    )
            
            load_end = datetime.utcnow()
            duration = (load_end - load_start).total_seconds()
            
            # Record load metadata
            load_metadata = {
                "load_id": load_id,
                "source_data": command.source_data,
                "target_connection": command.target_connection,
                "target_table": command.target_table,
                "load_mode": command.load_mode,
                "started_at": load_start.isoformat(),
                "completed_at": load_end.isoformat(),
                "duration_seconds": duration,
                "source_records": total_records,
                "loaded_records": load_stats.get("loaded_records", total_records),
                "batch_size": command.batch_size,
                "throughput_records_per_sec": total_records / duration if duration > 0 else 0,
                **load_stats
            }
            
            self.metadata_service.record_load(load_metadata)
            
            self.log_info("Data load completed successfully",
                         load_id=load_id,
                         loaded_records=load_stats.get("loaded_records", total_records),
                         duration_seconds=duration)
            
            return FlextResult[Dict[str, object]].ok(load_metadata)
            
        except Exception as e:
            self.log_error("Data load failed", load_id=load_id, error=str(e))
            return FlextResult[Dict[str, object]].fail(f"Load failed: {e}")
    
    def _execute_load_by_mode(self, 
                             connection, 
                             data: pd.DataFrame, 
                             command: LoadDataCommand,
                             load_id: str) -> FlextResult[Dict[str, object]]:
        """Execute loading based on the specified load mode."""
        try:
            if command.load_mode == "append":
                return self._append_load(connection, data, command, load_id)
            elif command.load_mode == "overwrite":
                return self._overwrite_load(connection, data, command, load_id)
            elif command.load_mode == "upsert":
                return self._upsert_load(connection, data, command, load_id)
            elif command.load_mode == "merge":
                return self._merge_load(connection, data, command, load_id)
            else:
                return FlextResult[Dict[str, object]].fail(f"Unsupported load mode: {command.load_mode}")
                
        except Exception as e:
            return FlextResult[Dict[str, object]].fail(f"Load execution failed: {e}")
    
    def _append_load(self, connection, data: pd.DataFrame, command: LoadDataCommand, load_id: str) -> FlextResult[Dict[str, object]]:
        """Execute append load."""
        try:
            loaded_records = 0
            batch_count = 0
            
            for start_idx in range(0, len(data), command.batch_size):
                end_idx = min(start_idx + command.batch_size, len(data))
                batch_data = data.iloc[start_idx:end_idx]
                
                result = connection.insert_batch(command.target_table, batch_data)
                if not result:
                    return FlextResult[Dict[str, object]].fail(f"Batch insert failed at index {start_idx}")
                
                batch_count += 1
                loaded_records += len(batch_data)
                
                self.log_info("Batch loaded",
                             load_id=load_id,
                             batch=batch_count,
                             records=len(batch_data),
                             total_loaded=loaded_records)
            
            return FlextResult[Dict[str, object]].ok({
                "loaded_records": loaded_records,
                "batch_count": batch_count,
                "load_mode": "append"
            })
            
        except Exception as e:
            return FlextResult[Dict[str, object]].fail(f"Append load failed: {e}")
    
    def _validate_post_load(self, connection, table: str, expected_count: int, load_mode: str) -> FlextResult[None]:
        """Validate data after loading."""
        try:
            # Check table exists
            if not connection.table_exists(table):
                return FlextResult[None].fail(f"Target table '{table}' not found after load")
            
            # Check record count for append mode
            if load_mode == "append":
                actual_count = connection.get_row_count(table)
                if actual_count < expected_count:
                    return FlextResult[None].fail(
                        f"Record count validation failed. Expected at least {expected_count}, got {actual_count}"
                    )
            
            # Additional validations can be added here
            
            return FlextResult[None].ok(None)
            
        except Exception as e:
            return FlextResult[None].fail(f"Post-load validation failed: {e}")
```

These examples demonstrate comprehensive CQRS patterns for ETL processing, providing structured operations for data extraction, transformation, and loading with proper error handling, quality gates, and monitoring capabilities.
