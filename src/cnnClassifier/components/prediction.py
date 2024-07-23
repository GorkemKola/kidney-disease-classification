import torch
from torchvision import transforms
from PIL import Image

class Prediction:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.transform = self.get_transform()
        self.classes = self.get_classes()
    def get_classes(self):
        classes = {
            0: 'Cyst',
            1: 'Normal',
            2: 'Stone',
            3: 'Tumor'
        }

        return classes

    def load_model(self):
        model = torch.load(self.config.best_model_path, map_location=self.device)
        model.eval()
        return model

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path) -> list[dict]:
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
        
        result = self.classes[predicted.item()]
        return [{'result':result}]

    def predict_batch(self, image_paths):
        images = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            image = self.transform(image)
            images.append(image)
        
        batch = torch.stack(images).to(self.device)
        
        
        with torch.no_grad():
            outputs = self.model(batch)
            _, predicted = torch.max(outputs, 1)
        predictions = predicted.cpu().numpy()
        predictions = [self.classes[pred.item()] for pred in predicted]
        return [{'result': pred} for pred in predictions]

