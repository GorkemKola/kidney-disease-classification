from cnnClassifier.constants import *
from cnnClassifier.utils import read_yaml, create_directories
from cnnClassifier.entity.config import DataIngestionConfig


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
    
