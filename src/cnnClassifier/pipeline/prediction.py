
from cnnClassifier.config import ConfigurationManager
from cnnClassifier.components.prediction import Prediction
from cnnClassifier import logger
from pathlib import Path
import glob

class PredictionPipeline:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.prediction_config = config_manager.get_prediction_config()
        self.predictor = Prediction(config=self.prediction_config)

    def predict(self, image_path):
        return self.predictor.predict(image_path)

    def batch_predict(self, image_dir):
        return self.predictor.predict_batch(image_dir)

STAGE_NAME = "Prediction"

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        pipeline = PredictionPipeline()
        
        # Single image prediction
        image_path = "artifacts/data_ingestion/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Normal/Normal- (2).jpg"
        predicted_class = pipeline.predict(image_path)
        logger.info(f"Predicted class for {image_path}: {predicted_class[0]['result']}")
        
        # Batch prediction
        image_paths = list(Path("artifacts/data_ingestion/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Stone").glob('*.jpg'))
        image_count = min(len(image_paths), 20)
        image_paths = image_paths[:image_count]

        predictions = pipeline.batch_predict(image_paths)
        for path, pred in zip(image_paths, predictions):
            logger.info(f"Predicted class for {path}: {pred['result']}")
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e