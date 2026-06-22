import datetime

from sqlalchemy.orm import selectinload

from src.database.schema import ExperimentStepModel, ModelSchemaModel, ModelLayerModel
from src.database.db_client import DBClient
from src.model_schema.model_schema_types import IModelSchema
from src.utils.logger.logger_interface import ILogger


class ExperimentStepModelService:
    def __init__(self, logger: ILogger):
        self._logger = logger
        self.db_client = DBClient()

    def _create_fingerprint(self, model_schema: IModelSchema):
        fingerprint = ""
        for layer in model_schema.layers:
            fingerprint += layer.type.value
            fingerprint += str(layer.units)
            fingerprint += layer.activation.value if layer.activation is not None else ""
            fingerprint += layer.regularizer.value if layer.regularizer is not None else ""

        fingerprint += model_schema.optimizer.value
        fingerprint += model_schema.loss.value

        return fingerprint

    def start_experiment_step(self, experiment_id: int, step: int, model_schema: IModelSchema):
        self._logger.log("Creating new step...", color="green")
        with self.db_client.session_scope() as session:
            new_step = ExperimentStepModel(
                experiment_id=experiment_id,
                fingerprint=self._create_fingerprint(model_schema),
                step=step,
            )
            session.add(new_step)
            session.flush()  # assign new_step.id before it is referenced as a foreign key

            schema = ModelSchemaModel(
                experiment_id=experiment_id,
                experiment_step_id=new_step.id,
                loss=model_schema.loss.value,
                optimizer=model_schema.optimizer.value,
            )
            session.add(schema)
            session.flush()  # assign schema.id before it is referenced as a foreign key

            session.add_all([ModelLayerModel(
                experiment_id=experiment_id,
                experiment_step_id=new_step.id,
                model_schema_id=schema.id,
                type=x.type.value,
                units=x.units,
                activation=x.activation.value if x.activation is not None else None,
                regularizer=x.regularizer.value if x.regularizer is not None else None,
            ) for x in model_schema.layers])

            session.commit()
            return new_step

    def finish_experiment_step(self, experiment_id: int, model_schema: IModelSchema, record_accuracy: float,
                               validation_accuracy: float):
        self._logger.log("Finishing step...", color="green")
        current = self.find(experiment_id, model_schema)
        if current is None:
            raise ValueError("Experiment step not found")
        with self.db_client.session_scope() as session:
            session.query(ExperimentStepModel).filter(ExperimentStepModel.id == current.id).update(
                {
                    ExperimentStepModel.endAt: datetime.datetime.now(),
                    ExperimentStepModel.record_accuracy: record_accuracy,
                    ExperimentStepModel.validation_accuracy: validation_accuracy,
                    ExperimentStepModel.accuracy_delta: (record_accuracy + validation_accuracy) / 2,
                })
            session.commit()
            return True

    def find(self, experiment_id: int, model_schema: IModelSchema):
        fingerprint = self._create_fingerprint(model_schema)
        with self.db_client.session_scope() as session:
            latest = session.query(ExperimentStepModel).where(ExperimentStepModel.experiment_id == experiment_id).where(
                ExperimentStepModel.fingerprint == fingerprint).first()
            return latest

    def get_schema(self, step_id: int):
        with self.db_client.session_scope() as session:
            schema = session.query(ModelSchemaModel).filter(
                ModelSchemaModel.experiment_step_id == step_id
            ).options(
                selectinload(ModelSchemaModel.model_layers)
            ).first()
            return schema

    def get_last_step(self, experiment_id: int):
        with self.db_client.session_scope() as session:
            latest = session.query(ExperimentStepModel).where(ExperimentStepModel.experiment_id == experiment_id).order_by(
                ExperimentStepModel.step.desc()).first()
            return latest.step if latest is not None else 1
