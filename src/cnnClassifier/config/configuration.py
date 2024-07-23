import os
from cnnClassifier.constants import *
from cnnClassifier.utils import read_yaml, create_directories
from cnnClassifier.entity.config import (
    DataIngestionConfig, 
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig
)

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH,
    ) -> None:
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([
            self.config.artifacts_root,
            self.config.data_ingestion.root_dir,
            self.config.data_ingestion.unzip_dir,
            self.config.prepare_base_model.root_dir
        ])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config 
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config

    def get_training_config(self):
        config = self.config.training
        prepare_base_model = self.config.prepare_base_model
        data_ingestion = self.config.data_ingestion

        params = self.params

        training_data = os.path.join(
            data_ingestion.unzip_dir,
            "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"    
        )

        training_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            last_model_path=Path(config.last_model_path),
            best_model_path=Path(config.best_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            tensorboard_log_dir=Path(config.tensorboard_log_dir),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_early_stopping_patience=params.EARLY_STOPPING_PATIENCE,
            params_learning_rate=params.LEARNING_RATE,
            params_random_state=params.RANDOM_STATE
        )
        
        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        evaluation_config = EvaluationConfig(
            path_of_model=Path(self.config.training.best_model_path),
            training_data=Path(self.config.data_ingestion.extracted_data),
            mlflow_uri=self.config.evaluation.mlflow_uri,
            params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_random_state=self.params.RANDOM_STATE
        )
        return evaluation_config