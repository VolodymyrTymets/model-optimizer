import tensorflow as tf
import datetime
from typing import List, Optional
from sqlalchemy.orm import selectinload, InstrumentedAttribute

from src.database.schema import ExperimentStepModel, ModelSchemaModel, ModelLayerModel, ImageModel, RecordResultModel, \
    WeightsModel
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
                               validation_accuracy: float, history: tf.keras.callbacks.History):
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
                    ExperimentStepModel.epochs: len(history.history['loss']),
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
            latest = session.query(ExperimentStepModel).where(
                ExperimentStepModel.experiment_id == experiment_id).order_by(
                ExperimentStepModel.step.desc()).first()
            return latest.step if latest is not None else 0

    def get_best_step(self, experiment_id: Optional[int] = None):
        with self.db_client.session_scope() as session:
            if experiment_id is None:
                return session.query(ExperimentStepModel).order_by(
                    ExperimentStepModel.accuracy_delta.desc()).first()
            if type(experiment_id) is not int:
                raise ValueError("Experiment id must be int or None")
            return session.query(ExperimentStepModel).where(
                ExperimentStepModel.experiment_id == experiment_id).order_by(
                ExperimentStepModel.accuracy_delta.desc()).first()

    def get_step(self, step_id: int):
        with self.db_client.session_scope() as session:
            return session.query(ExperimentStepModel).filter(ExperimentStepModel.id == step_id).first()

    def save_schema_plot(self, step_id: int, schema_plot: ImageModel):
        with self.db_client.session_scope() as session:
            session.add(schema_plot)
            schema = session.query(ModelSchemaModel).filter(ModelSchemaModel.experiment_step_id == step_id).first()
            if schema is None:
                return
            schema.plot = schema_plot
            session.flush()
            session.add(schema)
            session.commit()

    def save_training_history_plot(self, step_id: int, training_history_plot: ImageModel):
        with self.db_client.session_scope() as session:
            session.add(training_history_plot)
            step = session.query(ExperimentStepModel).filter(ExperimentStepModel.id == step_id).first()
            if step is None:
                return
            step.training_history_plot = training_history_plot
            session.flush()
            session.add(step)
            session.commit()

    def save_record_results(self, results: List[RecordResultModel]):
        with self.db_client.session_scope() as session:
            for result in results:
                session.add(result.image)
                session.flush()
                session.add(result)
                session.flush()
            session.commit()

    def get_weights(self, step_id: int) -> Optional[WeightsModel] :
        with self.db_client.session_scope() as session:
            step = session.query(ExperimentStepModel).filter(ExperimentStepModel.id == step_id).first()
            if step is None:
                return None
            if step.weights is None:
                return None
            return step.weights

    def save_final_weights(self, step_id: int, data: bytes):
        with self.db_client.session_scope() as session:
            step = session.query(ExperimentStepModel).filter(ExperimentStepModel.id == step_id).first()
            if step is None:
                return
            if step.weights is not None:
                step.weights.data = data
            else:
                weights = WeightsModel(data=data)
                session.add(weights)
                step.weights = weights
            session.flush()
            session.add(step)
            session.commit()
