from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """
    Abstract Base Class for all ML models in the serving platform.
    Any new model (YOLO, ResNet, etc.) must implement these methods.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Load model weights, configurations, and move to target device (CPU/GPU).
        """
        pass

    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """
        Convert raw input (e.g., bytes, base64, CV2 image) into the format
        required by the model (e.g., normalized tensors).
        """
        pass

    @abstractmethod
    def predict(self, model_input: Any) -> Any:
        """
        Core inference step (forward pass).
        Should ideally be framework-agnostic at the I/O boundary.
        """
        pass

    @abstractmethod
    def postprocess(self, model_output: Any) -> Any:
        """
        Convert raw model outputs (e.g., raw bounding box tensors)
        into structured, serializable formats (e.g., list of dicts).
        """
        pass

    def inference(self, input_data: Any) -> Any:
        """
        The complete end-to-end pipeline.
        Usually, you don't need to override this method in subclasses.
        """
        preprocessed_data = self.preprocess(input_data)
        predictions = self.predict(preprocessed_data)
        return self.postprocess(predictions)

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """
        Return model metadata for the API (name, version, task type, etc.).
        """
        pass
