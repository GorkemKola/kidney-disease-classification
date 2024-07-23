import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from cnnClassifier.utils import logger

class Training:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_base_model(self):
        self.model = torch.load(self.config.updated_base_model_path)
        self.model.to(self.device)

    def train_valid_test_loader(self):
        basic_transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if self.config.params_augmentation:
            train_transform = transforms.Compose([
                transforms.RandomRotation(40),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.config.params_image_size[0], scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                basic_transform
            ])
        else:
            train_transform = basic_transform

        full_dataset = datasets.ImageFolder(self.config.training_data, transform=train_transform)

        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)  # 70% for training
        valid_size = int(0.15 * total_size)  # 15% for validation
        test_size = total_size - train_size - valid_size  # 15% for testing

        # Ensure reproducibility
        generator = torch.Generator().manual_seed(self.config.params_random_state)

        train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size], generator=generator)

        # Apply transforms
        train_dataset.dataset.transform = train_transform
        valid_dataset.dataset.transform = basic_transform
        test_dataset.dataset.transform = basic_transform

        self.train_loader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=True, num_workers=4)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.config.params_batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.params_batch_size, shuffle=False, num_workers=4)

        logger.info(f"Number of training samples: {len(train_dataset)}")
        logger.info(f"Number of validation samples: {len(valid_dataset)}")
        logger.info(f"Number of test samples: {len(test_dataset)}")

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model, path)

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.params_learning_rate)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.config.params_learning_rate, steps_per_epoch=len(self.train_loader), epochs=self.config.params_epochs)
        best_valid_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(self.config.params_epochs):
            self.model.train()
            train_loss = 0.0
            train_pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.params_epochs} [Train]')
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item() * inputs.size(0)
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            train_loss = train_loss / len(self.train_loader.dataset)

            self.model.eval()
            valid_loss = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_predictions = []
            valid_pbar = tqdm(self.valid_loader, desc=f'Epoch {epoch+1}/{self.config.params_epochs} [Valid]')
            with torch.no_grad():
                for inputs, labels in valid_pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
                    
                    valid_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            valid_loss = valid_loss / len(self.valid_loader.dataset)
            accuracy = 100 * correct / total

            # Calculate precision, recall, and F1 score
            precision = precision_score(all_labels, all_predictions, average='weighted')
            recall = recall_score(all_labels, all_predictions, average='weighted')
            f1 = f1_score(all_labels, all_predictions, average='weighted')

            logger.info(f'Epoch {epoch+1}/{self.config.params_epochs}, '
                         f'Train Loss: {train_loss:.4f}, '
                         f'Valid Loss: {valid_loss:.4f}, '
                         f'Valid Accuracy: {accuracy:.2f}%, '
                         f'Precision: {precision:.4f}, '
                         f'Recall: {recall:.4f}, '
                         f'F1 Score: {f1:.4f}')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                early_stopping_counter = 0
                self.save_model(path=self.config.last_model_path, model=self.model)
                logger.info(f'Saved best model with valid loss: {valid_loss:.4f}')
                
                self.save_model(path=self.config.best_model_path, model=self.model)
                logger.info(f'Saved best model separately at: {self.config.best_model_path}')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.config.params_early_stopping_patience:
                    logger.info('Early stopping triggered')
                    break

        logger.info(f'Training completed. Best validation loss: {best_valid_loss:.4f}')
        logger.info(f'Best model saved at: {self.config.last_model_path}')
