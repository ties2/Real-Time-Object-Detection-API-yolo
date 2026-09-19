import pytest

from app.models.base import BaseModel
from app.models.registry import ModelRegistry


class FakeModel(BaseModel):

    def load(self) -> None:
        pass

    def predict(self, input_data):
        return input_data

    def metadata(self):
        return {
            "name": "fake",
            "version": "1.0.0",
        }

    def is_loaded(self) -> bool:
        return True


def test_register_and_get_model():
    registry = ModelRegistry()
    model = FakeModel()

    registry.register("fake", model)

    assert registry.exists("fake")
    assert registry.get("fake") is model


def test_duplicate_model_registration():
    registry = ModelRegistry()
    model = FakeModel()

    registry.register("fake", model)

    with pytest.raises(ValueError):
        registry.register("fake", model)


def test_list_models():
    registry = ModelRegistry()

    registry.register("fake", FakeModel())

    assert registry.list_models() == ["fake"]