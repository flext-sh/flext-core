from __future__ import annotations

# pyright: reportMissingImports=false

import sys
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import BaseModel, ValidationError

from flext_core import c, m, r, t, u


_base_module = sys.modules[m.Base.Metadata.__module__]


class _FrozenValue(m.Base.FrozenValueModel):
    name: str
    count: int


class _Identifiable(m.Base.IdentifiableMixin):
    pass


class _Timestampable(m.Base.TimestampableMixin):
    pass


class _Versionable(m.Base.VersionableMixin):
    pass


class _Auditable(m.Base.AuditableMixin):
    pass


class _SoftDelete(m.Base.SoftDeletableMixin):
    pass


class _Taggable(m.Base.TaggableMixin):
    pass


class _Validatable(m.Base.ValidatableMixin):
    value: int = 1


class _Serializable(m.Base.SerializableMixin):
    value: str
    optional: str | None = None


class _FailingValidatable(m.Base.ValidatableMixin):
    def validate_self(self) -> _FailingValidatable:
        msg = "invalid"
        raise ValueError(msg)


class _BrokenDumpModel(BaseModel):
    value: int = 1

    def __getattribute__(self, name: str) -> object:
        if name == "model_dump":
            return lambda *args, **kwargs: [1]
        return super().__getattribute__(name)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  value  ", "value"),
        ("name", "name"),
    ],
)
def test_strip_whitespace(raw: str, expected: str) -> None:
    assert _base_module.strip_whitespace(raw) == expected


def test_ensure_utc_datetime_naive_and_none() -> None:
    naive = datetime(2026, 1, 1, 12, 0, 0)
    assert _base_module.ensure_utc_datetime(naive).tzinfo == UTC
    assert _base_module.ensure_utc_datetime(None) is None


def test_normalize_to_list_branches() -> None:
    assert _base_module.normalize_to_list([1, 2]) == [1, 2]
    assert _base_module.normalize_to_list((1, 2)) == [1, 2]
    assert sorted(_base_module.normalize_to_list({1, 2})) == [1, 2]
    assert _base_module.normalize_to_list("x") == ["x"]


def test_validate_positive_number_via_field_constraint() -> None:
    """PositiveInt/PositiveFloat now use Field(gt=0) instead of custom validator."""
    from pydantic import TypeAdapter

    adapter = TypeAdapter(_base_module.PositiveInt)
    assert adapter.validate_python(1) == 1
    with pytest.raises(ValidationError):
        adapter.validate_python(0)
    with pytest.raises(ValidationError):
        adapter.validate_python(-1)

def test_validate_non_empty_string_success_and_error() -> None:
    assert _base_module.validate_non_empty_string("  data ") == "data"
    with pytest.raises(ValueError, match="String cannot be empty or whitespace"):
        _base_module.validate_non_empty_string("   ")


def test_validate_email_success_and_error() -> None:
    assert _base_module.validate_email("a@b.com") == "a@b.com"
    with pytest.raises(ValueError, match="Invalid email format"):
        _base_module.validate_email("invalid")


def test_validate_url_success_and_error() -> None:
    assert _base_module.validate_url("https://example.com") == "https://example.com"
    with pytest.raises(ValueError, match="Invalid URL format"):
        _base_module.validate_url("example")


def test_validate_semver_success_and_error() -> None:
    assert _base_module.validate_semver("1.2.3-alpha+001") == "1.2.3-alpha+001"
    with pytest.raises(ValueError, match="Invalid semantic version format"):
        _base_module.validate_semver("1.2")


def test_validate_uuid_string_success_and_error() -> None:
    valid = str(uuid.uuid4())
    assert _base_module.validate_uuid_string(valid) == valid
    with pytest.raises(ValueError, match="Invalid UUID format"):
        _base_module.validate_uuid_string("nope")


def test_validate_config_dict_errors_and_reserved_key() -> None:
    with pytest.raises(TypeError, match="Configuration must be a dictionary"):
        _base_module.validate_config_dict("x")
    with pytest.raises(ValueError, match="reserved"):
        _base_module.validate_config_dict({"_secret": 1})


def test_validate_tags_list_normalization_and_errors() -> None:
    assert _base_module.validate_tags_list([" A ", "a", "B", " "]) == ["a", "b"]
    with pytest.raises(TypeError, match="Tags must be a list"):
        _base_module.validate_tags_list("x")
    with pytest.raises(TypeError, match="Tag must be string"):
        _base_module.validate_tags_list([1])


def test_metadata_attributes_accepts_none() -> None:
    model = m.Base.Metadata(attributes=None)
    assert model.attributes == {}


def test_metadata_attributes_accepts_basemodel_mapping() -> None:
    class _Attrs(BaseModel):
        key: str

    model = m.Base.Metadata(attributes=_Attrs(key="value"))
    assert model.attributes == {"key": "value"}


def test_metadata_attributes_rejects_basemodel_non_mapping_dump() -> None:
    with pytest.raises(TypeError, match="must dump to mapping"):
        m.Base.Metadata(attributes=_BrokenDumpModel())


def test_metadata_attributes_accepts_t_dict_and_mapping() -> None:
    model_from_t_dict = m.Base.Metadata(attributes=t.Dict(root={"a": 1}))
    model_from_mapping = m.Base.Metadata(attributes={"b": 2})
    assert model_from_t_dict.attributes == {"a": 1}
    assert model_from_mapping.attributes == {"b": 2}


def test_metadata_attributes_t_dict_branch_when_basemodel_check_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NotPydanticBase:
        pass

    monkeypatch.setattr(_base_module, "BaseModel", _NotPydanticBase)
    assert m.Base.Metadata._validate_attributes(t.Dict(root={"x": 1})) == {"x": 1}


def test_metadata_attributes_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="attributes must be dict-like"):
        m.Base.Metadata(attributes=123)


def test_frozen_value_model_equality_and_hash() -> None:
    left = _FrozenValue(name="item", count=1)
    right = _FrozenValue(name="item", count=1)
    assert left == right
    assert left.__eq__(object()) is NotImplemented
    assert isinstance(hash(left), int)


def test_identifiable_unique_id_validation_and_computed_fields() -> None:
    model = _Identifiable(unique_id="prefix-12345678")
    assert model.id_short == "prefix-1"
    assert model.id_prefix == "prefix"
    assert model.is_uuid_format is False
    assert isinstance(model.id_hash, int)


def test_identifiable_prefix_none_without_separator() -> None:
    model = _Identifiable(unique_id="abcdef12")
    assert model.id_prefix is None


def test_identifiable_uuid_format_true_and_regeneration() -> None:
    original_uuid = str(uuid.uuid4())
    model = _Identifiable(unique_id=original_uuid)
    previous = model.unique_id
    model.regenerate_id()
    assert model.unique_id != previous
    assert model.validate_id_consistency() is model
    assert model.id_prefix == model.unique_id.split("-", 1)[0]
    assert model.is_uuid_format is True


def test_identifiable_unique_id_empty_rejected() -> None:
    with pytest.raises(ValidationError, match="String should have at least 1 character"):
        _Identifiable(unique_id="   ")


def test_timestampable_timestamp_conversion_and_json_serializer() -> None:
    naive = datetime(2026, 1, 1, 12, 0, 0)
    model = _Timestampable(created_at=naive, updated_at=naive)
    assert model.created_at.tzinfo == UTC
    dumped = model.model_dump(mode="json")
    assert isinstance(dumped["created_at"], str)


def test_timestampable_age_and_recent_flags() -> None:
    created_at = datetime.now(UTC) - timedelta(minutes=10)
    model = _Timestampable(created_at=created_at)
    assert model.is_modified is False
    assert model.age_seconds > 0
    assert model.age_minutes > 0
    assert model.age_hours > 0
    assert model.age_days > 0
    assert model.last_modified_age_seconds is None
    assert model.is_recent is True
    assert model.is_very_recent is False


def test_timestampable_formatted_age_branches() -> None:
    model = _Timestampable(
        created_at=datetime.now(UTC) - timedelta(days=1, hours=2, minutes=3)
    )
    formatted = model.time_since_creation_formatted
    assert "d" in formatted
    assert "h" in formatted
    assert "m" in formatted


def test_timestampable_touch_and_consistency_error() -> None:
    model = _Timestampable()
    model.touch()
    assert model.updated_at is not None
    with pytest.raises(ValidationError, match="updated_at cannot be before created_at"):
        _Timestampable(
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_versionable_computed_fields_and_mutation_methods() -> None:
    model = _Versionable(version=2)
    assert model.is_initial_version is False
    assert model.version_string == "v2"
    assert model.is_even_version is True
    assert model.is_odd_version is False
    model.increment_version()
    assert model.version == 3
    model.set_version(10)
    assert model.version == 10
    model.reset_to_initial_version()
    assert model.version == c.Performance.DEFAULT_VERSION


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (c.Performance.DEFAULT_VERSION, "initial"),
        (c.Performance.VERSION_LOW_THRESHOLD, "low"),
        (c.Performance.VERSION_MEDIUM_THRESHOLD, "medium"),
        (c.Performance.VERSION_MEDIUM_THRESHOLD + 1, "high"),
    ],
)
def test_versionable_version_category(version: int, expected: str) -> None:
    assert _Versionable(version=version).version_category == expected


def test_versionable_validation_errors() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        _Versionable(version=-1)
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        _Versionable(version=0)
    model = _Versionable(version=1)
    with pytest.raises(ValueError, match="Version must be >="):
        model.set_version(0)


def test_versionable_internal_validators_for_unreachable_branches() -> None:
    # validate_version field_validator removed; ge=MIN_VERSION handles it declaratively
    raw = _Versionable.model_construct(version=0)
    with pytest.raises(ValueError, match="below minimum allowed"):
        raw.validate_version_consistency()


def test_auditable_user_and_timestamp_validators() -> None:
    model = _Auditable(
        created_by="  creator  ",
        created_at=datetime(2026, 1, 1, 12, 0, 0),
        updated_at=datetime(2026, 1, 1, 13, 0, 0),
        updated_by=" updater ",
    )
    assert model.created_by == "creator"
    assert model.updated_by == "updater"
    assert model.created_at.tzinfo == UTC
    dumped = model.model_dump(mode="json")
    assert isinstance(dumped["created_at"], str)


def test_auditable_computed_fields_and_mutation_methods() -> None:
    model = _Auditable(created_by="alice")
    assert model.has_audit_info is True
    assert model.was_modified_by_different_user is False
    assert model.audit_summary == "created by alice"
    model.set_created_by("bob")
    assert model.created_by == "bob"
    model.set_updated_by("charlie")
    assert model.updated_by == "charlie"
    assert model.updated_at is not None
    model.audit_update("dora")
    assert model.updated_by == "dora"
    assert "updated by dora" in model.audit_summary


def test_auditable_validation_errors() -> None:
    with pytest.raises(ValidationError, match="String should have at least 1 character"):
        _Auditable(created_by="   ")
    with pytest.raises(ValidationError, match="updated_at set but updated_by is None"):
        _Auditable(created_by="x", updated_at=datetime.now(UTC))
    with pytest.raises(ValidationError, match="created_by must be set"):
        _Auditable()


def test_soft_delete_validators_and_serializer() -> None:
    model = _SoftDelete(
        deleted_by=" user ",
        deleted_at=datetime(2026, 1, 1, 12, 0, 0),
        is_deleted=True,
    )
    assert model.deleted_by == "user"
    assert model.deleted_at is not None and model.deleted_at.tzinfo == UTC
    dumped = model.model_dump(mode="json")
    assert isinstance(dumped["deleted_at"], str)


def test_soft_delete_states_and_restore_cycle() -> None:
    model = _SoftDelete()
    assert model.is_active is True
    assert model.can_be_restored is False

    with pytest.raises(ValidationError, match="deleted_at is set but is_deleted=False"):
        model.soft_delete("deleter")

    deleted = _SoftDelete(
        is_deleted=True,
        deleted_at=datetime.now(UTC),
        deleted_by="deleter",
    )
    assert deleted.can_be_restored is True

    with pytest.raises(ValidationError, match="deleted_at is set but is_deleted=False"):
        deleted.restore()

    model = _SoftDelete()
    model.restore()
    assert model.is_deleted is False
    assert model.deleted_at is None
    assert model.deleted_by is None


def test_soft_delete_validation_errors() -> None:
    with pytest.raises(ValidationError, match="String should have at least 1 character"):
        _SoftDelete(deleted_by="   ")
    with pytest.raises(ValidationError, match="is_deleted=True but deleted_at is None"):
        _SoftDelete(is_deleted=True)
    with pytest.raises(ValidationError, match="deleted_at is set but is_deleted=False"):
        _SoftDelete(deleted_at=datetime.now(UTC), is_deleted=False)


def test_taggable_validators_and_computed_fields() -> None:
    model = _Taggable(
        tags=[" one ", "one", "two"],
        categories=[" alpha ", "alpha", "beta"],
        labels={" x ": " y ", "": "skip", "k": ""},
    )
    assert model.tags == ["one", "two"]
    assert model.categories == ["alpha", "beta"]
    assert model.labels == {"x": "y"}
    assert model.tag_count == 2
    assert model.category_count == 2
    assert model.has_tags is True
    assert model.has_categories is True
    assert model.all_labels == ["x"]


def test_taggable_collection_methods() -> None:
    model = _Taggable(tags=["one"], categories=["cat"], labels={"k": "v"})
    model.add_tag("one")
    model.add_tag("two")
    assert model.tags == ["one", "two"]
    model.remove_tag("one")
    assert model.tags == ["two"]
    assert model.has_tag("two") is True

    model.add_category("cat")
    model.add_category("new")
    assert model.categories == ["cat", "new"]
    model.remove_category("cat")
    assert model.categories == ["new"]
    assert model.has_category("new") is True

    model.set_label("owner", "team")
    assert model.get_label("owner") == "team"
    assert model.get_label("missing", "default") == "default"
    model.remove_label("owner")
    assert model.get_label("owner") is None


def test_taggable_overlap_validation_error() -> None:
    with pytest.raises(ValidationError, match="overlapping keys"):
        _Taggable(tags=["dup"], labels={"dup": "1"})


def test_validatable_default_behaviors() -> None:
    model = _Validatable()
    assert model.validate_business_rules() is model
    assert model.validate_self().value == 1
    assert model.is_valid() is True
    assert model.get_validation_errors() == []


def test_validatable_failure_paths() -> None:
    model = _FailingValidatable()
    assert model.is_valid() is False
    errors = model.get_validation_errors()
    assert len(errors) == 1
    assert "invalid" in errors[0]


def test_serializable_mixin_methods() -> None:
    model = _Serializable(value="x")
    as_dict = model.to_dict()
    as_json = model.to_json(indent=2)
    assert as_dict == {"value": "x"}
    assert '"value": "x"' in as_json
    rebuilt = _Serializable.from_dict({"value": "x", "optional": None})
    from_json = _Serializable.from_json('{"value":"y"}')
    assert rebuilt.value == "x"
    assert from_json.value == "y"


def test_advanced_serializable_methods() -> None:
    model = m.Base.AdvancedSerializable(
        name="sample",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        metadata=t.Dict(root={"n": 1}),
    )
    dumped = model.model_dump(mode="json")
    assert dumped["timestamp"].startswith("2026-01-01T")
    assert dumped["metadata"] == {"n": "1"}
    formats = model.to_json_multiple_formats()
    assert "iso_timestamps" in formats
    assert "compact" in formats
    assert "unix_timestamp" in formats
    assert model.formatted_name == "[SAMPLE]"


def test_dynamic_rebuild_model_methods() -> None:
    model = m.Base.DynamicRebuildModel(name="x", value=5)
    assert model.doubled_value == 10

    extra_cls = m.Base.DynamicRebuildModel.create_with_extra_field("extra", int)
    assert "extra" in extra_cls.__annotations__

    with pytest.raises(ValueError, match="has no field"):
        model.add_runtime_field("runtime_key", "v")
    assert model.get_runtime_field("missing", "default") == "default"

    plus_one_cls = m.Base.DynamicRebuildModel.rebuild_with_validator(
        lambda value: int(value) + 1
    )
    plus_one = plus_one_cls(name="x", value=2)
    assert plus_one.value == 3


def test_dynamic_model_methods() -> None:
    model = m.Base.DynamicModel.create_dynamic("dyn", a=1)
    assert model.name == "dyn"
    assert model.dynamic_field_count == 1
    assert model.has_dynamic_fields is True
    model.add_field("b", 2)
    assert model.fields.root == {"a": 1, "b": 2}
    rebuilt = model.rebuild_with_validation()
    assert rebuilt.fields.root == {"a": 1, "b": 2}


def test_timestamped_model_and_alias_and_canonical_symbols() -> None:
    model = m.Base.TimestampedModel()
    assert model.created_at.tzinfo == UTC
    assert m.Base is m.Base
    assert r[str].ok("ok").value == "ok"
    assert c.Performance.DEFAULT_VERSION >= 1
    assert hasattr(u, "mapper")
