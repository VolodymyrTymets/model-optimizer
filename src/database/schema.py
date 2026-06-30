import datetime
from typing import List
from typing import Optional
from sqlalchemy import Table
from sqlalchemy import Column
import sqlalchemy as sa
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    id: Mapped[int] = mapped_column(primary_key=True)
    createdAt: Mapped[datetime.datetime] = mapped_column(sa.DateTime(timezone=True), default=datetime.datetime.now)


results_on_experiment_step_association_table = Table(
    "results_on_experiment_step_association_table",
    Base.metadata,
    Column("experiment_step_id", ForeignKey("experiment_step.id"), primary_key=True),
    Column("image_id", ForeignKey("image.id"), primary_key=True),
)


class ExperimentModel(Base):
    __tablename__ = "experiment"

    endAt: Mapped[datetime.datetime] = mapped_column(sa.DateTime(timezone=True), nullable=True)

    details: Mapped["ExperimentDetailsModel"] = relationship(back_populates="experiment")
    data_set_details: Mapped["ExperimentDataSetDetailsModel"] = relationship(back_populates="experiment")
    steps: Mapped[List["ExperimentStepModel"]] = relationship(back_populates="experiment")
    model_schemas: Mapped[List["ModelSchemaModel"]] = relationship(back_populates="experiment")
    model_layers: Mapped[List["ModelLayerModel"]] = relationship(back_populates="experiment")

    def __repr__(self) -> str:
        return f"<ExperimentModel(id={self.id!r}), endAt={self.endAt!r}>"


class ExperimentDetailsModel(Base):
    __tablename__ = "experiment_details"

    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment.id"))
    layers: Mapped[str] = mapped_column(sa.String)
    activation: Mapped[str] = mapped_column(sa.String)
    optimizer: Mapped[str] = mapped_column(sa.String)
    regularizer: Mapped[str] = mapped_column(sa.String, nullable=True)
    loss: Mapped[str] = mapped_column(sa.String)
    epochs: Mapped[int] = mapped_column(sa.Integer)
    batch_size: Mapped[int] = mapped_column(sa.Integer)
    units_range: Mapped[str] = mapped_column(sa.String)

    experiment: Mapped["ExperimentModel"] = relationship(back_populates="details")

    def __repr__(self):
        return f"ExperimentDetailsModel(epochs={int(self.epochs)}, batch_size={int(self.batch_size)}, layers={self.layers}, activation={self.activation}, optimizer={self.optimizer}, regularizer={self.regularizer}, loss={self.loss}, units_range={self.units_range})"


class ExperimentDataSetDetailsModel(Base):
    __tablename__ = "experiment_data_set_details"
    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment.id"))
    duration: Mapped[float] = mapped_column(sa.Float)
    labels: Mapped[str] = mapped_column(sa.String)
    argumentation_types: Mapped[str] = mapped_column(sa.String)
    af_type: Mapped[str] = mapped_column(sa.String)

    experiment: Mapped["ExperimentModel"] = relationship(back_populates="data_set_details")


class ExperimentStepModel(Base):
    __tablename__ = "experiment_step"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment.id"))
    endAt: Mapped[datetime.datetime] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    step: Mapped[int] = mapped_column(sa.Integer)
    fingerprint: Mapped[str] = mapped_column(sa.String)
    record_accuracy: Mapped[float] = mapped_column(sa.Float, default=0)
    validation_accuracy: Mapped[float] = mapped_column(sa.Float, default=0)
    accuracy_delta: Mapped[float] = mapped_column(sa.Float, default=0)
    epochs: Mapped[int] = mapped_column(sa.Integer, default=0)
    training_history_plot_id: Mapped[Optional[int]] = mapped_column(ForeignKey("image.id"), nullable=True)

    schema: Mapped["ModelSchemaModel"] = relationship(back_populates="step")
    experiment: Mapped["ExperimentModel"] = relationship(back_populates="steps")
    training_history_plot: Mapped[Optional["ImageModel"]] = relationship(back_populates="training_history_plots")
    results: Mapped[List["ImageModel"]] = relationship(secondary=results_on_experiment_step_association_table)


class ModelSchemaModel(Base):
    __tablename__ = "model_schema"

    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment.id"))
    experiment_step_id: Mapped[int] = mapped_column(ForeignKey("experiment_step.id"))
    optimizer: Mapped[str] = mapped_column(sa.String)
    loss: Mapped[str] = mapped_column(sa.String)
    model_layers: Mapped[List["ModelLayerModel"]] = relationship(back_populates="model_schema")

    step: Mapped["ExperimentStepModel"] = relationship(back_populates="schema")
    experiment: Mapped["ExperimentModel"] = relationship(back_populates="model_schemas")
    plot_id: Mapped[Optional[int]] = mapped_column(ForeignKey("image.id"), nullable=True)
    plot: Mapped[Optional["ImageModel"]] = relationship(back_populates="model_schemas")

    def __repr__(self):
        return f"<ModelSchemaModel(id={self.id}, optimizer={self.optimizer}, loss={self.loss})>"


class ModelLayerModel(Base):
    __tablename__ = "model_layer"
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment.id"))
    experiment_step_id: Mapped[int] = mapped_column(ForeignKey("experiment_step.id"))
    model_schema_id: Mapped[int] = mapped_column(ForeignKey("model_schema.id"))
    type: Mapped[str] = mapped_column(sa.String)
    units: Mapped[int] = mapped_column(sa.Integer)
    activation: Mapped[str] = mapped_column(sa.String, nullable=True)
    regularizer: Mapped[str] = mapped_column(sa.String, nullable=True)

    experiment: Mapped["ExperimentModel"] = relationship(back_populates="model_layers")
    model_schema: Mapped["ModelSchemaModel"] = relationship(back_populates="model_layers")


class ImageModel(Base):
    __tablename__ = "image"
    id: Mapped[int] = mapped_column(primary_key=True)
    base64: Mapped[str] = mapped_column(sa.String)

    training_history_plots: Mapped[List["ExperimentStepModel"]] = relationship(back_populates="training_history_plot")
    model_schemas: Mapped[List["ModelSchemaModel"]] = relationship(back_populates="plot")
    result_images: Mapped[List["ExperimentStepModel"]] =  relationship(secondary=results_on_experiment_step_association_table)
