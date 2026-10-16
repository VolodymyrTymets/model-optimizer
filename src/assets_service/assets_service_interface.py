from abc import ABC, abstractmethod
from typing import Optional


class IAssetsService(ABC):
    @abstractmethod
    def get_assets_path(self) -> str:
        pass

    @abstractmethod
    def get_data_set_path(self) -> str:
        pass

    @abstractmethod
    def get_experiment_path(self) -> str:
        pass

    @abstractmethod
    def get_validation_records_path(self) -> str:
        pass

    @abstractmethod
    def get_model_path(self) -> str:
        pass

    def get_models_path(self, model_name: Optional[str] = None) -> tuple[str, str]:
        pass

    @abstractmethod
    def get_data_set_path_for_model(self, model_name: str) -> str:
        pass