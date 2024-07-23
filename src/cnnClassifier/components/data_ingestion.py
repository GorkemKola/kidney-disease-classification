import os
import zipfile
import gdown
from cnnClassifier.utils import logger, get_size
from cnnClassifier.entity import DataIngestionConfig


class DataIngestion:
    def __init__(
            self,
            config: DataIngestionConfig
        ) -> None:
        self.config = config

    def download_file(self):
        '''
        Description
            fetch data from the url
        '''

        
        try:
            dataset_url = self.config.source_url
            local_data_file = self.config.local_data_file

            logger.info(f'Downloading data from {dataset_url} into file {local_data_file}')
            file_id = dataset_url.split('/')[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id, local_data_file)
            logger.info(f'Downloaded data from {dataset_url} into file {local_data_file}')
        except Exception as e:
            raise e
        
    def extract_zip_file(self):
        '''
        Description
            extracts the zip file
        '''

        unzip_path = self.config.unzip_dir

        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        
        logger.info(f'{self.config.local_data_file} is successfully extracted to {unzip_path}')