import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import mlflow
import json
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import dagshub
from config import REPO_NAME, AUTHOR_USER_NAME

class Evaluation:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_valid_test_loader(self):
        basic_transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        full_dataset = datasets.ImageFolder(self.config.training_data, transform=basic_transform)

        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)  # 70% for training
        valid_size = int(0.15 * total_size)  # 15% for validation
        test_size = total_size - train_size - valid_size  # 15% for testing

        # Ensure reproducibility
        generator = torch.Generator().manual_seed(self.config.params_random_state)

        *_, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size], generator=generator)

        test_dataset.dataset.transform = basic_transform

        self.test_loader = DataLoader(test_dataset, batch_size=self.config.params_batch_size, shuffle=False, num_workers=4)


    @staticmethod
    def load_model(path: Path) -> torch.nn.Module:
        model = torch.load(path)
        model.eval()  # Set the model to evaluation mode
        return model

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self.train_valid_test_loader()
        self.loss, self.accuracy, self.precision, self.recall, self.f1 = self.evaluate_model()
        self.save_score()
        self.log_into_mlflow()

    def evaluate_model(self):
        self.model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        # Use tqdm to create a progress bar for evaluation
        valid_pbar = tqdm(self.test_loader, desc='Evaluating', unit='batch')
        
        with torch.no_grad():
            for inputs, labels in valid_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                valid_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        loss = total_loss / len(self.test_loader.dataset)
        accuracy = correct / total

        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )

        return loss, accuracy, precision, recall, f1

    def save_score(self):
        scores = {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1
        }
        with open(Path("scores.json"), 'w') as f:
            json.dump(scores, f, indent=4)

    def log_into_mlflow(self):
        dagshub.init(repo_owner=AUTHOR_USER_NAME, repo_name=REPO_NAME, mlflow=True)

        with mlflow.start_run():

            mlflow.log_params(self.config.params)
            mlflow.log_metrics(
                {
                    "loss": self.loss,
                    "accuracy": self.accuracy,
                    "precision": self.precision,
                    "recall": self.recall,
                    "f1_score": self.f1
                }
            )
            # Log model directly if model registry is not applicable
            mlflow.pytorch.log_model(self.model, "model")