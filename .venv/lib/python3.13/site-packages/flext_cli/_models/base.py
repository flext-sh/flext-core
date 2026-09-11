"""CLI Pydantic domain models."""

from __future__ import annotations

from flext_cli._models._base.flextclimodelsbase_part_01 import (
    FlextCliModelsBase as FlextCliModelsBasePart01,
)
from flext_cli._models._base.flextclimodelsbase_part_02 import (
    FlextCliModelsBase as FlextCliModelsBasePart02,
)
from flext_cli._models._base.flextclimodelsbase_part_03 import (
    FlextCliModelsBase as FlextCliModelsBasePart03,
)
from flext_cli._models._base.flextclimodelsbase_part_04 import (
    FlextCliModelsBase as FlextCliModelsBasePart04,
)
from flext_cli._models._base.flextclimodelsbase_part_05 import (
    FlextCliModelsBase as FlextCliModelsBasePart05,
)
from flext_cli._models._base.flextclimodelsbase_part_06 import (
    FlextCliModelsBase as FlextCliModelsBasePart06,
)
from flext_cli._models._base.flextclimodelsbase_part_07 import (
    FlextCliModelsBase as FlextCliModelsBasePart07,
)


class FlextCliModelsBase(
    FlextCliModelsBasePart01,
    FlextCliModelsBasePart02,
    FlextCliModelsBasePart03,
    FlextCliModelsBasePart04,
    FlextCliModelsBasePart05,
    FlextCliModelsBasePart06,
    FlextCliModelsBasePart07,
):
    """Public facade for FlextCliModelsBase."""


__all__: list[str] = ["FlextCliModelsBase"]
