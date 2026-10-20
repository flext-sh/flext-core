from tests import m
import pytest

def test_debug8():
    print("ConfigMap root:", m.ConfigMap.__pydantic_core_schema__["schema"]["values_schema"])
    model = m.Metadata(attributes={"source": "nested"})
    try:
        m.ConfigMap.model_validate({"items": [model]})
        print("ACCEPTED")
    except Exception as e:
        print("REJECTED:", type(e).__name__)
