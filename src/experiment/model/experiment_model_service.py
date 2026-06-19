import datetime

from src.database.schema import ExperimentModel as DBExperimentModel, ExperimentDetailsModel
from src.database.db_client import DBClient
from src.experiment.experiment_types import IExperimentDetails, ExperimentDetails
from src.model_schema.model_schema_types import LayerType, ActivationType, OptimizerType, RegularizerType, LossType
from src.utils.logger.logger_interface import ILogger


class ExperimentModelService:
    def __init__(self, logger: ILogger):
        self._logger = logger
        self.db_client = DBClient()

    def _create(self, experiment_details: IExperimentDetails):
        self._logger.log("Creating new experiment...")
        with self.db_client.session_scope() as session:
            details = ExperimentDetailsModel(
                layers=','.join(x.value for x in experiment_details.layers),
                activation=','.join(x.value for x in experiment_details.activation),
                optimizer=','.join(x.value for x in experiment_details.optimizer),
                regularizer=','.join(x.value for x in experiment_details.regularizer),
                loss=','.join(x.value for x in experiment_details.loss),
                epochs=experiment_details.epochs,
                batch_size=experiment_details.batch_size,
                units_range=','.join([str(x) for x in experiment_details.units_range]),
            )
            new_experiment = DBExperimentModel(
                details=details
            )
            session.add_all([new_experiment])
            session.commit()
            return new_experiment

    def finish_experiment(self, experiment_id: int):
        with self.db_client.session_scope() as session:
            session.query(DBExperimentModel).filter(DBExperimentModel.id == experiment_id).update(
                {DBExperimentModel.endAt: datetime.datetime.now()})
            session.commit()
            return True

    def get_current_experiment(self, experiment_details: IExperimentDetails):
        with self.db_client.session_scope() as session:
            latest = session.query(DBExperimentModel).order_by(DBExperimentModel.id.desc()).first()
            if latest is not None and latest.endAt is None:
                self._logger.log("Found not finished experiment", color="yellow")
                return latest
            return self._create(experiment_details)


    def get_details(self, experiment: DBExperimentModel) -> IExperimentDetails:
        with self.db_client.session_scope() as session:
            details = session.query(ExperimentDetailsModel).filter(DBExperimentModel.id == experiment.id).first()
            return ExperimentDetails(
                epochs=details.epochs,
                batch_size=details.batch_size,
                layers=[LayerType(x) for x in details.layers.split(',')],
                activation=[ActivationType(x) for x in details.activation.split(',')],
                units_range=[int(x) for x in details.units_range.split(',')],
                optimizer=[OptimizerType(x) for x in details.optimizer.split(',')],
                regularizer=[RegularizerType(x) for x in details.regularizer.split(',')],
                loss=[LossType(x) for x in details.loss.split(',')],
            )
