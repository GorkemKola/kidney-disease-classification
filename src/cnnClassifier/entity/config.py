from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_include_top: bool
    params_learning_rate: float
    params_weights: str
    params_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    last_model_path: Path
    best_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_augmentation: bool
    params_image_size: list
    params_early_stopping_patience: int
    params_learning_rate: float
    params_random_state: int


@dataclass
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
    params_random_state: int

@dataclass(frozen=True)
class PredictionConfig:
    root_dir: Path
    best_model_path: Path
    params_image_size: list
    params_batch_size: int
    params_classes: int