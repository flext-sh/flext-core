"""CLI protocol-backed composite aliases."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path

from ruamel.yaml.comments import CommentedMap, CommentedSeq
from tomlkit.container import Container
from tomlkit.items import AoT, Array, Item, Table
from tomlkit.toml_document import TOMLDocument

from flext_cli import c
from flext_cli._typings.base import FlextCliTypesBase as tb
from flext_core import p, t

# NOTE (multi-agent): YAML round-trip aliases live at module level because a
# self-referencing PEP 695 alias in CLASS scope breaks pyrefly/pyright (proven
# in cosmos-charts). The class body only re-exports them as ``t.Cli.Yaml*``;
# never inline the recursion into the class body.
type _YamlScalar = str | int | float | bool | None
type _YamlValue = _YamlScalar | list[_YamlValue] | Mapping[str, _YamlValue]
type _YamlNode = CommentedMap | CommentedSeq | _YamlScalar
type _YamlSequence = CommentedSeq | list[_YamlValue]


class FlextCliTypesDomain:
    """Composite CLI aliases built from canonical protocols and core types."""

    type YamlScalar = _YamlScalar
    type YamlValue = _YamlValue
    type YamlNode = _YamlNode
    type YamlSequence = _YamlSequence

    type ResultValue = t.JsonPayload
    type RuleDefinitions = t.SequenceOf[t.JsonMapping]
    type RuleMatcher = tuple[
        frozenset[str], frozenset[str], frozenset[str], frozenset[str]
    ]
    type RuleMatchers = t.SequenceOf[RuleMatcher]
    type RuleCatalog[TKind] = t.MappingKV[
        TKind,
        t.SequenceOf[
            tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str]]
        ],
    ]
    type MatchedRuleDefinition[TKind] = tuple[TKind, t.JsonMapping]
    type MatchedRuleDefinitions[TKind] = t.SequenceOf[tuple[TKind, t.JsonMapping]]
    type RuleLoadResult[TRuleKind, TFileRuleKind] = tuple[
        t.SequenceOf[tuple[TRuleKind, t.JsonMapping]],
        t.SequenceOf[tuple[TFileRuleKind, t.JsonMapping]],
    ]
    type RuleLoadOption[TRuleKind, TFileRuleKind] = (
        tb.CliValue
        | Path
        | FlextCliTypesDomain.RuleCatalog[TRuleKind]
        | FlextCliTypesDomain.RuleCatalog[TFileRuleKind]
        | None
    )
    type MutableDefaultMapping = MutableMapping[str, t.Scalar | t.StrSequence]
    type CliParamValue = bool | str
    type CliParamKwargs = t.MappingKV[str, CliParamValue]
    type DefaultAtom = t.Scalar | t.StrSequence
    type ProjectNamesValue = str | t.StrSequence
    type TableHeaders = str | t.StrSequence
    type IntTextValue = int | str
    type MessageType = c.Cli.MessageTypes
    type ModelLike = t.BaseModel
    # mro-j47u (codex): model classes use the canonical core type alias.
    type ModelSource = ModelLike | t.JsonMapping | t.ScalarMapping
    type OptionRegistry = t.MappingKV[str, t.MappingKV[str, t.Scalar | t.StrSequence]]
    type NullaryOperation[T] = Callable[[], T]
    type PromptTextReader = Callable[[str], str]
    type CliCommand = Callable[..., t.JsonPayload]
    # mro-j47u (codex): one generic alias owns formatter data and call contracts.
    type JsonCommandFn = Callable[..., p.Result[t.JsonPayload]]
    type SuccessMessageFormatter[TResult: ResultValue = ResultValue] = Callable[
        [TResult], str
    ]
    type MappingProcessor[T, U] = Callable[[str, T], U]
    type TomlMappingSource = (
        t.JsonPayload | t.JsonMapping | t.ScalarMapping | Item | TOMLDocument
    )
    type TomlUnwrappedSource = t.JsonPayload | t.JsonMapping | Item
    type TomlStringListSource = (
        TomlUnwrappedSource | t.SequenceOf[t.JsonPayload] | t.SequenceOf[t.Primitives]
    )
    type TomlRuntimeSource = (
        TOMLDocument
        | Table
        | Item
        | Array
        | AoT
        | Container
        | t.JsonMapping
        | t.JsonPayload
    )
    type TypeKind = c.Cli.TypeKind
    type TypedExtractValue = str | bool | t.JsonMapping
    type TableDataSource = tb.TabularData | t.SequenceOf[t.JsonMapping]
    type TextPath = str | Path
    type JsonWriteData = t.JsonPayload


__all__: list[str] = ["FlextCliTypesDomain"]
