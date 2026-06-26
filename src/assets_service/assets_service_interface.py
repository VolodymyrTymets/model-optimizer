from abc import ABC, abstractmethod


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