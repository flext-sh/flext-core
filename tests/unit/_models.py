from __future__ import annotations

from typing import TypeAlias

from . import _models_impl as impl


class TestUnitModels:
    _ValidationLikeError = impl._ValidationLikeError
    TestCaseMap: TypeAlias = impl.TestCaseMap
    InputPayloadMap: TypeAlias = impl.InputPayloadMap
    _MsgWithCommandId = impl._MsgWithCommandId
    _MsgWithMessageId = impl._MsgWithMessageId
    SampleModel = impl.SampleModel
    _SvcModel = impl._SvcModel
    _BrokenDumpModel = impl._BrokenDumpModel
    _ErrorsModel = impl._ErrorsModel
    _PlainErrorModel = impl._PlainErrorModel
    _TargetModel = impl._TargetModel
    CacheTestModel = impl.CacheTestModel
    NestedModel = impl.NestedModel
    ConfigModelForTest = impl.ConfigModelForTest
    InvalidModelForTest = impl.InvalidModelForTest
    SingletonClassForTest = impl.SingletonClassForTest
    BadConfigForTest = impl.BadConfigForTest
    _DumpErrorModel = impl._DumpErrorModel
    _Opts = impl._Opts
    _FakeConfig = impl._FakeConfig
    _Model = impl._Model
    _SampleEntity = impl._SampleEntity
    _FrozenEntity = impl._FrozenEntity
    _GoodModel = impl._GoodModel
    ComplexModel = impl.ComplexModel
    _Cfg = impl._Cfg
    _BadCopyModel = impl._BadCopyModel


__all__ = ["TestUnitModels"]
