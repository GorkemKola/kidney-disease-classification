from cnnClassifier.config import ConfigurationManager
from cnnClassifier.components.training import Training
from cnnClassifier import logger

class TrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        print(training_config)
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_test_loader()
        training.train()

STAGE_NAME = 'Training Stage'

if __name__ == '__main__':
    try:
        logger.info(
            f'>>>>> stage {STAGE_NAME} started <<<<<'
        )
        obj = TrainingPipeline()
        obj.main()
        logger.info(
              f'>>>>> stage {STAGE_NAME} completed <<<<<\n\nx===========x'
        )

    except Exception as e:
        logger.info(e)
        raise e
