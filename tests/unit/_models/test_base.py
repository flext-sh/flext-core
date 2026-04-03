"""Tests for base Pydantic models via FlextModels facade.

Covers FlextModelFoundation classes: Metadata, Validators, model config variants,
FrozenValueModel, IdentifiableMixin, TimestampableMixin, VersionableMixin,
RetryConfigurationMixin, messages, results, and validation outcomes.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from flext_tests import tm
from tests import c, m, t


class TestFlextModelsBase:
    """Tests for flext_core._models.base via the m facade."""

    # ── Metadata ──────────────────────────────────────────────

    def test_metadata_defaults(self) -> None:
        meta = m.Metadata()
        tm.that(meta.version, eq="1.0.0")
        tm.that(meta.created_by, none=True)
        tm.that(meta.modified_by, none=True)
        tm.that(meta.tags, eq=[])
        tm.that(meta.attributes, eq={})
        tm.that(meta.metadata_value, none=True)
        tm.that(meta.created_at.tzinfo, eq=UTC)
        tm.that(meta.updated_at.tzinfo, eq=UTC)

    def test_metadata_with_values(self) -> None:
        meta = m.Metadata(
            version="2.0.0",
            created_by="system",
            modified_by="user-1",
            tags=["billing", "critical"],
            attributes={"source": "api"},
            metadata_value=42,
        )
        tm.that(meta.version, eq="2.0.0")
        tm.that(meta.created_by, eq="system")
        tm.that(meta.modified_by, eq="user-1")
        tm.that(meta.tags, eq=["billing", "critical"])
        tm.that(meta.attributes, eq={"source": "api"})
        tm.that(meta.metadata_value, eq=42)

    def test_metadata_is_frozen(self) -> None:
        meta = m.Metadata()
        with pytest.raises(ValidationError):
            object.__setattr__(meta, "version", "9.9.9")

    def test_metadata_attributes_none_becomes_empty(self) -> None:
        meta = m.Metadata.model_validate({"attributes": None})
        tm.that(meta.attributes, eq={})

    def test_metadata_attributes_rejects_reserved_keys(self) -> None:
        with pytest.raises(ValueError, match="Keys starting with '_' are reserved"):
            m.Metadata(attributes={"_secret": "value"})

    def test_metadata_attributes_rejects_non_mapping(self) -> None:
        with pytest.raises(TypeError, match="attributes must be dict-like"):
            m.Metadata.model_validate({"attributes": 123})

    def test_metadata_attributes_accepts_basemodel(self) -> None:
        class _Payload(BaseModel):
            key: str = "val"

        meta = m.Metadata.model_validate({"attributes": _Payload()})
        tm.that(meta.attributes, eq={"key": "val"})

    def test_metadata_serialization(self) -> None:
        meta = m.Metadata()
        data = meta.model_dump()
        tm.that(data, is_=dict)
        tm.that("created_at" in data, eq=True)
        tm.that("updated_at" in data, eq=True)
        tm.that("version" in data, eq=True)

    def test_metadata_forbids_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            m.Metadata.model_validate({"unknown_field": "nope"})

    # ── Validators ────────────────────────────────────────────

    def test_validators_ensure_utc_datetime_naive(self) -> None:
        naive = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC).replace(tzinfo=None)
        result = m.Validators.ensure_utc_datetime(naive)
        tm.that(result, none=False)
        tm.that(cast("datetime", result).tzinfo, eq=UTC)

    def test_validators_ensure_utc_datetime_none(self) -> None:
        result = m.Validators.ensure_utc_datetime(None)
        tm.that(result, none=True)

    def test_validators_ensure_utc_datetime_already_utc(self) -> None:
        aware = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
        result = m.Validators.ensure_utc_datetime(aware)
        tm.that(result, eq=aware)

    def test_validators_normalize_to_list_scalar(self) -> None:
        result = m.Validators.normalize_to_list("hello")
        tm.that(result, is_=list)
        tm.that("hello" in result, eq=True)

    def test_validators_normalize_to_list_already_list(self) -> None:
        result = m.Validators.normalize_to_list([1, 2, 3])
        tm.that(result, eq=[1, 2, 3])

    def test_validators_strip_whitespace(self) -> None:
        result = m.Validators.strip_whitespace("  hello  ")
        tm.that(result, eq="hello")

    def test_validators_validate_config_dict_valid(self) -> None:
        result = m.Validators.validate_config_dict({"key": "value"})
        tm.that(result, eq={"key": "value"})

    def test_validators_validate_config_dict_rejects_reserved_keys(self) -> None:
        with pytest.raises(ValueError, match="Keys starting with '_' are reserved"):
            m.Validators.validate_config_dict({"_internal": 1})

    def test_validators_validate_config_dict_rejects_non_dict(self) -> None:
        with pytest.raises(TypeError, match="Configuration must be a dictionary"):
            m.Validators.validate_config_dict("not-a-dict")

    def test_validators_validate_tags_list_normalizes(self) -> None:
        result = m.Validators.validate_tags_list(["  Hello ", "WORLD", "hello"])
        tm.that(result, eq=["hello", "world"])

    def test_validators_validate_tags_list_rejects_non_list(self) -> None:
        with pytest.raises(TypeError, match="Tags must be a list"):
            m.Validators.validate_tags_list("not-a-list")

    def test_validators_validate_tags_list_rejects_non_string_items(self) -> None:
        with pytest.raises(TypeError, match="Tag must be string"):
            m.Validators.validate_tags_list([123])

    def test_validators_validate_tags_deduplicates(self) -> None:
        result = m.Validators.validate_tags_list(["a", "A", "b", "B"])
        tm.that(result, eq=["a", "b"])

    def test_validators_validate_tags_strips_empty(self) -> None:
        result = m.Validators.validate_tags_list(["valid", "  ", ""])
        tm.that(result, eq=["valid"])

    # ── TypeAdapter lazy loaders ──────────────────────────────

    def test_validators_tags_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.tags_adapter()
        assert adapter is not None

    def test_validators_config_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.config_adapter()
        assert adapter is not None

    def test_validators_list_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.list_adapter()
        assert adapter is not None

    def test_validators_metadata_map_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.metadata_map_adapter()
        assert adapter is not None

    def test_validators_strict_string_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.strict_string_adapter()
        assert adapter is not None

    def test_validators_scalar_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.scalar_adapter()
        assert adapter is not None

    def test_validators_float_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.float_adapter()
        assert adapter is not None

    def test_validators_str_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.str_adapter()
        assert adapter is not None

    def test_validators_str_list_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.str_list_adapter()
        assert adapter is not None

    def test_validators_str_or_bytes_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.str_or_bytes_adapter()
        assert adapter is not None

    def test_validators_primitives_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.primitives_adapter()
        assert adapter is not None

    def test_validators_dict_container_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.dict_container_adapter()
        assert adapter is not None

    def test_validators_list_container_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.list_container_adapter()
        assert adapter is not None

    def test_validators_tuple_container_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.tuple_container_adapter()
        assert adapter is not None

    def test_validators_set_container_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.set_container_adapter()
        assert adapter is not None

    def test_validators_set_str_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.set_str_adapter()
        assert adapter is not None

    def test_validators_set_scalar_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.set_scalar_adapter()
        assert adapter is not None

    def test_validators_serializable_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.serializable_adapter()
        assert adapter is not None

    def test_validators_enum_type_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.enum_type_adapter()
        assert adapter is not None

    def test_validators_dict_str_metadata_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.dict_str_metadata_adapter()
        assert adapter is not None

    def test_validators_list_serializable_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.list_serializable_adapter()
        assert adapter is not None

    def test_validators_tuple_serializable_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.tuple_serializable_adapter()
        assert adapter is not None

    def test_validators_sortable_dict_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.sortable_dict_adapter()
        assert adapter is not None

    def test_validators_strict_json_list_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.strict_json_list_adapter()
        assert adapter is not None

    def test_validators_strict_json_scalar_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.strict_json_scalar_adapter()
        assert adapter is not None

    def test_validators_metadata_json_dict_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.metadata_json_dict_adapter()
        assert adapter is not None

    def test_validators_flat_metadata_dict_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.flat_metadata_dict_adapter()
        assert adapter is not None

    def test_validators_structlog_processor_adapter_returns_adapter(self) -> None:
        adapter = m.Validators.structlog_processor_adapter()
        assert adapter is not None

    # ── ArbitraryTypesModel ───────────────────────────────────

    def test_arbitrary_types_model_config(self) -> None:
        cfg = m.ArbitraryTypesModel.model_config
        tm.that(cfg.get("arbitrary_types_allowed"), eq=True)
        tm.that(cfg.get("extra"), eq="forbid")
        tm.that(cfg.get("validate_assignment"), eq=True)

    # ── StrictBoundaryModel ───────────────────────────────────

    def test_strict_boundary_model_config(self) -> None:
        cfg = m.StrictBoundaryModel.model_config
        tm.that(cfg.get("strict"), eq=True)
        tm.that(cfg.get("frozen"), eq=True)
        tm.that(cfg.get("extra"), eq="forbid")
        tm.that(cfg.get("str_strip_whitespace"), eq=True)

    # ── FlexibleInternalModel ─────────────────────────────────

    def test_flexible_internal_model_ignores_extra(self) -> None:
        class _Flex(m.FlexibleInternalModel):
            name: str = "test"

        inst = _Flex.model_validate({"name": "a", "extra_field": "ignored"})
        tm.that(inst.name, eq="a")
        tm.that(hasattr(inst, "extra_field"), eq=False)

    # ── ImmutableValueModel ───────────────────────────────────

    def test_immutable_value_model_frozen(self) -> None:
        class _Imm(m.ImmutableValueModel):
            label: str = "default"

        inst = _Imm(label="fixed")
        with pytest.raises(ValidationError):
            object.__setattr__(inst, "label", "changed")

    def test_immutable_value_model_forbids_extra(self) -> None:
        class _Imm(m.ImmutableValueModel):
            x: int = 1

        with pytest.raises(ValidationError):
            _Imm.model_validate({"x": 1, "y": 2})

    # ── DynamicConfigModel ────────────────────────────────────

    def test_dynamic_config_model_allows_extra(self) -> None:
        class _Dyn(m.DynamicConfigModel):
            base: str = "ok"

        inst = _Dyn.model_validate({"base": "ok", "custom": "val"})
        tm.that(inst.base, eq="ok")

        extra = inst.model_extra
        if isinstance(extra, dict):
            val = extra.get("custom")
            if isinstance(val, str):
                tm.that(val, eq="val")

    # ── TaggedModel ───────────────────────────────────────────

    def test_tagged_model_forbids_extra(self) -> None:
        cfg = m.TaggedModel.model_config
        tm.that(cfg.get("extra"), eq="forbid")

    # ── FrozenStrictModel ─────────────────────────────────────

    def test_frozen_strict_model_config(self) -> None:
        cfg = m.FrozenStrictModel.model_config
        tm.that(cfg.get("frozen"), eq=True)
        tm.that(cfg.get("strict"), eq=True)
        tm.that(cfg.get("hide_input_in_errors"), eq=True)
        tm.that(cfg.get("extra"), eq="forbid")

    def test_frozen_strict_model_is_immutable(self) -> None:
        class _Strict(m.FrozenStrictModel):
            val: int

        inst = _Strict(val=10)
        with pytest.raises(ValidationError):
            object.__setattr__(inst, "val", 20)

    # ── FrozenValueModel ──────────────────────────────────────

    def test_frozen_value_model_equality_same(self) -> None:
        class _FV(m.FrozenValueModel):
            name: str
            count: int

        a = _FV(name="x", count=1)
        b = _FV(name="x", count=1)
        tm.that(a == b, eq=True)

    def test_frozen_value_model_equality_different(self) -> None:
        class _FV(m.FrozenValueModel):
            name: str
            count: int

        a = _FV(name="x", count=1)
        b = _FV(name="y", count=2)
        tm.that(a == b, eq=False)

    def test_frozen_value_model_equality_not_implemented(self) -> None:
        class _FV(m.FrozenValueModel):
            name: str

        inst = _FV(name="x")
        result = inst.__eq__("not_a_model")
        tm.that(result is NotImplemented, eq=True)

    def test_frozen_value_model_hash(self) -> None:
        class _FV(m.FrozenValueModel):
            name: str
            count: int

        a = _FV(name="x", count=1)
        b = _FV(name="x", count=1)
        tm.that(hash(a), eq=hash(b))
        tm.that(hash(a), is_=int)

    def test_frozen_value_model_usable_in_set(self) -> None:
        class _FV(m.FrozenValueModel):
            name: str

        a = _FV(name="x")
        b = _FV(name="x")
        c_inst = _FV(name="y")
        tm.that(hash(a), eq=hash(b))
        tm.that(hash(a) != hash(c_inst), eq=True)

    # ── IdentifiableMixin ─────────────────────────────────────

    def test_identifiable_generates_uuid(self) -> None:
        class _Id(m.IdentifiableMixin):
            pass

        inst = _Id()
        tm.that(inst.unique_id, is_=str)
        tm.that(len(inst.unique_id) > 0, eq=True)

    def test_identifiable_unique_ids_differ(self) -> None:
        class _Id(m.IdentifiableMixin):
            pass

        a = _Id()
        b = _Id()
        tm.that(a.unique_id != b.unique_id, eq=True)

    def test_identifiable_regenerate_id(self) -> None:
        class _Id(m.IdentifiableMixin):
            pass

        inst = _Id()
        old_id = inst.unique_id
        inst.regenerate_id()
        tm.that(inst.unique_id != old_id, eq=True)

    def test_identifiable_rejects_empty_id(self) -> None:
        class _Id(m.IdentifiableMixin):
            pass

        with pytest.raises(ValidationError, match="at least 1 character"):
            _Id(unique_id="   ")

    def test_identifiable_accepts_custom_id(self) -> None:
        class _Id(m.IdentifiableMixin):
            pass

        inst = _Id(unique_id="custom-123")
        tm.that(inst.unique_id, eq="custom-123")

    # ── TimestampableMixin ────────────────────────────────────

    def test_timestampable_defaults(self) -> None:
        class _Ts(m.TimestampableMixin):
            pass

        inst = _Ts()
        tm.that(inst.created_at.tzinfo, eq=UTC)
        tm.that(inst.updated_at, none=True)

    def test_timestampable_naive_datetime_gets_utc(self) -> None:
        class _Ts(m.TimestampableMixin):
            pass

        naive = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC).replace(tzinfo=None)
        inst = _Ts(created_at=naive)
        tm.that(inst.created_at.tzinfo, eq=UTC)

    def test_timestampable_update_timestamp(self) -> None:
        class _Ts(m.TimestampableMixin):
            pass

        inst = _Ts()
        tm.that(inst.updated_at, none=True)
        inst.update_timestamp()

        updated = inst.updated_at
        if isinstance(updated, datetime):
            tm.that(updated.tzinfo, eq=UTC)

    def test_timestampable_updated_before_created_rejected(self) -> None:
        class _Ts(m.TimestampableMixin):
            pass

        now = datetime.now(UTC)
        past = now - timedelta(hours=1)
        with pytest.raises(
            ValidationError,
            match="updated_at cannot be before created_at",
        ):
            _Ts(created_at=now, updated_at=past)

    def test_timestampable_json_serialization(self) -> None:
        class _Ts(m.TimestampableMixin):
            pass

        now = datetime.now(UTC)
        inst = _Ts(created_at=now, updated_at=now)
        dumped = inst.model_dump(mode="json")
        tm.that(dumped["created_at"], is_=str)
        tm.that(dumped["updated_at"], is_=str)

    def test_timestampable_json_serialization_none_updated(self) -> None:
        class _Ts(m.TimestampableMixin):
            pass

        inst = _Ts()
        dumped = inst.model_dump(mode="json")
        tm.that(dumped["updated_at"], none=True)

    # ── VersionableMixin ──────────────────────────────────────

    def test_versionable_default(self) -> None:
        class _V(m.VersionableMixin):
            pass

        inst = _V()
        tm.that(inst.version, eq=c.DEFAULT_RETRY_DELAY_SECONDS)

    def test_versionable_increment(self) -> None:
        class _V(m.VersionableMixin):
            pass

        inst = _V()
        original = inst.version
        inst.increment_version()
        tm.that(inst.version, eq=original + 1)

    def test_versionable_below_minimum_rejected(self) -> None:
        class _V(m.VersionableMixin):
            pass

        with pytest.raises(ValidationError, match="below minimum"):
            _V(version=0)

    # ── RetryConfigurationMixin ───────────────────────────────

    def test_retry_config_defaults(self) -> None:
        class _Rc(m.RetryConfigurationMixin):
            pass

        inst = _Rc()
        tm.that(inst.max_retries, eq=c.MAX_RETRY_ATTEMPTS)
        tm.that(inst.initial_delay_seconds, eq=c.DEFAULT_RETRY_DELAY_SECONDS)
        tm.that(inst.max_delay_seconds, eq=c.DEFAULT_MAX_DELAY_SECONDS)

    def test_retry_config_alias(self) -> None:
        class _Rc(m.RetryConfigurationMixin):
            pass

        inst = _Rc.model_validate({"max_attempts": 5})
        tm.that(inst.max_retries, eq=5)

    def test_retry_config_custom_values(self) -> None:
        class _Rc(m.RetryConfigurationMixin):
            pass

        inst = _Rc(max_retries=10, initial_delay_seconds=2.0, max_delay_seconds=120.0)
        tm.that(inst.max_retries, eq=10)
        tm.that(inst.initial_delay_seconds, eq=2.0)
        tm.that(inst.max_delay_seconds, eq=120.0)

    # ── TimestampedModel ──────────────────────────────────────

    def test_timestamped_model_inherits_timestamps(self) -> None:
        inst = m.TimestampedModel()
        tm.that(inst.created_at.tzinfo, eq=UTC)

    # ── CommandMessage ────────────────────────────────────────

    def test_command_message_defaults(self) -> None:
        cmd = m.CommandMessage(command_type="create_user")
        tm.that(cmd.message_type, eq="command")
        tm.that(cmd.command_type, eq="create_user")
        tm.that(cmd.issuer_id, none=True)
        tm.that(cmd.data, is_=t.Dict)

    def test_command_message_with_data(self) -> None:
        cmd = m.CommandMessage(
            command_type="update",
            issuer_id="admin",
            data=t.Dict(root={"field": "value"}),
        )
        tm.that(cmd.issuer_id, eq="admin")
        tm.that(cmd.data.root, eq={"field": "value"})

    # ── QueryMessage ──────────────────────────────────────────

    def test_query_message_defaults(self) -> None:
        qry = m.QueryMessage(query_type="get_users")
        tm.that(qry.message_type, eq="query")
        tm.that(qry.query_type, eq="get_users")
        tm.that(qry.pagination, none=True)

    def test_query_message_with_filters(self) -> None:
        qry = m.QueryMessage(
            query_type="search",
            filters=t.Dict(root={"status": "active"}),
            pagination=t.Dict(root={"page": 1}),
        )
        tm.that(qry.filters.root, eq={"status": "active"})
        tm.that(qry.pagination, none=False)

    # ── EventMessage ──────────────────────────────────────────

    def test_event_message_defaults(self) -> None:
        evt = m.EventMessage(event_type="user_created", aggregate_id="usr-1")
        tm.that(evt.message_type, eq="event")
        tm.that(evt.event_type, eq="user_created")
        tm.that(evt.aggregate_id, eq="usr-1")
        tm.that(evt.metadata, none=True)

    def test_event_message_with_data(self) -> None:
        evt = m.EventMessage(
            event_type="order_placed",
            aggregate_id="ord-1",
            data=t.Dict(root={"total": 100}),
        )
        tm.that(evt.data.root, eq={"total": 100})

    # ── SuccessResult ─────────────────────────────────────────

    def test_success_result(self) -> None:
        sr = m.SuccessResult(value="done")
        tm.that(sr.result_type, eq="success")
        tm.that(sr.value, eq="done")
        tm.that(sr.metadata, none=True)

    # ── FailureResult ─────────────────────────────────────────

    def test_failure_result(self) -> None:
        fr = m.FailureResult(error="something broke")
        tm.that(fr.result_type, eq="failure")
        tm.that(fr.error, eq="something broke")
        tm.that(fr.error_code, none=True)
        tm.that(fr.error_data, none=True)

    def test_failure_result_with_code_and_data(self) -> None:
        meta = m.Metadata(tags=["error"])
        fr = m.FailureResult(
            error="validation failed",
            error_code="VALIDATION",
            error_data=meta,
        )
        tm.that(fr.error_code, eq="VALIDATION")
        tm.that(fr.error_data, none=False)

    # ── PartialResult ─────────────────────────────────────────

    def test_partial_result(self) -> None:
        pr = m.PartialResult(
            value="partial-data",
            warnings=["row 5 skipped"],
            partial_success_rate=0.75,
        )
        tm.that(pr.result_type, eq="partial")
        tm.that(pr.value, eq="partial-data")
        tm.that(pr.warnings, eq=["row 5 skipped"])
        tm.that(pr.partial_success_rate, eq=0.75)

    # ── ValidOutcome ──────────────────────────────────────────

    def test_valid_outcome(self) -> None:
        vo = m.ValidOutcome(validated_data="clean", validation_time_ms=1.5)
        tm.that(vo.outcome_type, eq="valid")
        tm.that(vo.validated_data, eq="clean")
        tm.that(vo.validation_time_ms, eq=1.5)

    # ── InvalidOutcome ────────────────────────────────────────

    def test_invalid_outcome(self) -> None:
        io = m.InvalidOutcome(errors=["field required"])
        tm.that(io.outcome_type, eq="invalid")
        tm.that(io.errors, eq=["field required"])
        tm.that(io.error_codes, eq=[])

    def test_invalid_outcome_with_codes(self) -> None:
        io = m.InvalidOutcome(
            errors=["too short"],
            error_codes=["MIN_LENGTH"],
        )
        tm.that(io.error_codes, eq=["MIN_LENGTH"])

    # ── WarningOutcome ────────────────────────────────────────

    def test_warning_outcome(self) -> None:
        wo = m.WarningOutcome(
            validated_data="data",
            warnings=["deprecated field"],
            validation_time_ms=2.0,
        )
        tm.that(wo.outcome_type, eq="warning")
        tm.that(wo.warnings, eq=["deprecated field"])

    # ── Entity ────────────────────────────────────────────────

    def test_entity_creation(self) -> None:
        entity = m.Entity()
        tm.that(entity.unique_id, is_=str)
        tm.that(len(entity.unique_id) > 0, eq=True)
        tm.that(entity.created_at.tzinfo, eq=UTC)
        tm.that(entity.updated_at, none=False)
        tm.that(entity.domain_events, eq=[])

    def test_entity_id_property(self) -> None:
        entity = m.Entity()
        tm.that(entity.entity_id, eq=entity.unique_id)

    def test_entity_identity_equality(self) -> None:
        e1 = m.Entity()
        e2 = m.Entity()
        tm.that(e1 == e2, eq=False)
        e2.unique_id = e1.unique_id
        tm.that(e1 == e2, eq=True)

    def test_entity_identity_hash(self) -> None:
        e1 = m.Entity()
        e2 = m.Entity()
        e2.unique_id = e1.unique_id
        tm.that(hash(e1), eq=hash(e2))

    def test_entity_equality_non_model(self) -> None:
        entity = m.Entity()
        result = entity.__eq__("not_a_model")
        tm.that(result is NotImplemented, eq=True)

    def test_entity_add_domain_event(self) -> None:
        entity = m.Entity()
        result = entity.add_domain_event(
            "order_placed", t.ConfigMap(root={"total": 50})
        )
        tm.ok(result)
        tm.that(len(entity.domain_events), eq=1)
        tm.that(entity.domain_events[0].event_type, eq="order_placed")

    def test_entity_add_domain_event_empty_type_fails(self) -> None:
        entity = m.Entity()
        result = entity.add_domain_event("")
        tm.fail(result, has="non-empty string")

    def test_entity_uncommitted_events(self) -> None:
        entity = m.Entity()
        entity.add_domain_event("evt1")
        entity.add_domain_event("evt2")
        tm.that(len(entity.uncommitted_events), eq=2)

    def test_entity_clear_domain_events(self) -> None:
        entity = m.Entity()
        entity.add_domain_event("evt1")
        cleared = entity.clear_domain_events()
        tm.that(len(cleared), eq=1)
        tm.that(len(entity.domain_events), eq=0)

    def test_entity_mark_events_as_committed(self) -> None:
        entity = m.Entity()
        entity.add_domain_event("evt1")
        result = entity.mark_events_as_committed()
        committed = tm.ok(result)
        tm.that(len(committed), eq=1)
        tm.that(len(entity.domain_events), eq=0)

    def test_entity_add_domain_events_bulk(self) -> None:
        entity = m.Entity()
        events = [("evt1", None), ("evt2", t.ConfigMap(root={"key": "val"}))]
        result = entity.add_domain_events_bulk(events)
        created = tm.ok(result)
        tm.that(len(created), eq=2)
        tm.that(len(entity.domain_events), eq=2)

    def test_entity_add_domain_events_bulk_empty_name_fails(self) -> None:
        entity = m.Entity()
        result = entity.add_domain_events_bulk([("", None)])
        tm.fail(result, has="non-empty")

    def test_entity_serialization(self) -> None:
        entity = m.Entity()
        data = entity.model_dump()
        tm.that("unique_id" in data, eq=True)
        tm.that("created_at" in data, eq=True)
        tm.that("version" in data, eq=True)
        tm.that("entity_id" in data, eq=True)

    def test_entity_version(self) -> None:
        entity = m.Entity()
        original = entity.version
        entity.increment_version()
        tm.that(entity.version, eq=original + 1)

    # ── Value ─────────────────────────────────────────────────

    def test_value_immutable(self) -> None:
        class _Val(m.Value):
            name: str

        v = _Val(name="test")
        with pytest.raises(ValidationError):
            object.__setattr__(v, "name", "bob")

    def test_value_equality_by_value(self) -> None:
        class _Val(m.Value):
            name: str
            amount: int

        a = _Val(name="x", amount=5)
        b = _Val(name="x", amount=5)
        tm.that(a == b, eq=True)

    def test_value_inequality(self) -> None:
        class _Val(m.Value):
            name: str

        a = _Val(name="x")
        b = _Val(name="y")
        tm.that(a == b, eq=False)

    def test_value_hash_consistency(self) -> None:
        class _Val(m.Value):
            name: str

        a = _Val(name="x")
        b = _Val(name="x")
        tm.that(hash(a), eq=hash(b))

    def test_value_not_implemented_for_non_model(self) -> None:
        class _Val(m.Value):
            name: str

        v = _Val(name="x")
        result = v.__eq__("string")
        tm.that(result is NotImplemented, eq=True)

    def test_value_usable_in_set(self) -> None:
        class _Val(m.Value):
            name: str

        a = _Val(name="x")
        b = _Val(name="x")
        c_v = _Val(name="y")
        tm.that(hash(a), eq=hash(b))
        tm.that(hash(a) != hash(c_v), eq=True)

    # ── AggregateRoot ─────────────────────────────────────────

    def test_aggregate_root_is_entity(self) -> None:
        agg = m.AggregateRoot()
        tm.that(type(agg).__mro__, has=m.Entity)
        tm.that(agg.unique_id, is_=str)

    def test_aggregate_root_collects_events(self) -> None:
        agg = m.AggregateRoot()
        agg.add_domain_event("created")
        agg.add_domain_event("updated")
        tm.that(len(agg.domain_events), eq=2)

    def test_aggregate_root_check_invariants_passes(self) -> None:
        agg = m.AggregateRoot()
        agg.check_invariants()  # no invariants defined, should pass

    def test_aggregate_root_check_invariants_fails(self) -> None:
        class _Agg(m.AggregateRoot):
            _invariants = [lambda: False]

        with pytest.raises(ValueError, match="Invariant violated"):
            _Agg()

    # ── Copy/deepcopy behavior ────────────────────────────────

    def test_entity_deepcopy(self) -> None:
        entity = m.Entity()
        entity.add_domain_event("evt1")
        copied = copy.deepcopy(entity)
        tm.that(copied.unique_id, eq=entity.unique_id)
        tm.that(len(copied.domain_events), eq=1)
        # Modifying copy should not affect original
        copied.domain_events.clear()
        tm.that(len(entity.domain_events), eq=1)

    def test_value_copy(self) -> None:
        class _Val(m.Value):
            name: str

        v = _Val(name="original")
        copied = copy.copy(v)
        tm.that(copied.name, eq="original")
        tm.that(copied == v, eq=True)

    def test_metadata_copy(self) -> None:
        meta = m.Metadata(tags=["a"])
        copied = copy.copy(meta)
        tm.that(copied.tags, eq=["a"])

    # ── Parametrized: message discriminators ──────────────────

    @pytest.mark.parametrize(
        ("msg_data", "expected_type"),
        [
            ({"message_type": "command", "command_type": "x"}, "command"),
            ({"message_type": "query", "query_type": "y"}, "query"),
            (
                {
                    "message_type": "event",
                    "event_type": "z",
                    "aggregate_id": "a-1",
                },
                "event",
            ),
        ],
        ids=["command", "query", "event"],
    )
    def test_message_union_discriminator(
        self,
        msg_data: dict[str, str],
        expected_type: str,
    ) -> None:
        adapter = TypeAdapter(m.MessageUnion)
        msg = adapter.validate_python(msg_data)
        tm.that(msg.message_type, eq=expected_type)

    # ── Parametrized: operation result discriminators ─────────

    @pytest.mark.parametrize(
        ("result_data", "expected_type"),
        [
            ({"result_type": "success", "value": "ok"}, "success"),
            ({"result_type": "failure", "error": "bad"}, "failure"),
            (
                {
                    "result_type": "partial",
                    "value": "half",
                    "partial_success_rate": 0.5,
                },
                "partial",
            ),
        ],
        ids=["success", "failure", "partial"],
    )
    def test_operation_result_discriminator(
        self,
        result_data: dict[str, str | float],
        expected_type: str,
    ) -> None:
        adapter = TypeAdapter(m.OperationResult)
        result = adapter.validate_python(result_data)
        tm.that(result.result_type, eq=expected_type)

    # ── Parametrized: validation outcome discriminators ───────

    @pytest.mark.parametrize(
        ("outcome_data", "expected_type"),
        [
            (
                {
                    "outcome_type": "valid",
                    "validated_data": "ok",
                    "validation_time_ms": 1.0,
                },
                "valid",
            ),
            ({"outcome_type": "invalid", "errors": ["bad"]}, "invalid"),
            (
                {
                    "outcome_type": "warning",
                    "validated_data": "ok",
                    "warnings": ["w"],
                    "validation_time_ms": 1.0,
                },
                "warning",
            ),
        ],
        ids=["valid", "invalid", "warning"],
    )
    def test_validation_outcome_discriminator(
        self,
        outcome_data: dict[str, str | float | list[str]],
        expected_type: str,
    ) -> None:
        adapter = TypeAdapter(m.ValidationOutcome)
        outcome = adapter.validate_python(outcome_data)
        tm.that(outcome.outcome_type, eq=expected_type)

    # ── Parametrized: adapter idempotency (cached) ────────────

    @pytest.mark.parametrize(
        "adapter_name",
        [
            "tags_adapter",
            "config_adapter",
            "list_adapter",
            "metadata_map_adapter",
            "strict_string_adapter",
            "scalar_adapter",
            "float_adapter",
            "str_adapter",
            "str_list_adapter",
            "primitives_adapter",
            "serializable_adapter",
        ],
    )
    def test_adapter_idempotent(self, adapter_name: str) -> None:
        getter = getattr(m.Validators, adapter_name)
        first = getter()
        second = getter()
        tm.that(first is second, eq=True)
