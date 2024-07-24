# Kidney Disease Classification

### Abstract

This project classifies diseases in Kidney CT images using a fine-tuned ResNet-50 model. The dataset, sourced from Kaggle, consists of images labeled into one of four categories: Cyst, Normal, Stone, and Tumor. To address this classification problem, a production-level pipeline was designed utilizing DVC and CI/CD methodologies. The data pipeline comprises four main components: Data Ingestion, Base Model Preparation, Training, and Evaluation. Additionally, a Flask-based web app is implemented for inference, allowing users to upload their CT images and receive real-time classification results.

### Requirements

- Python 3.12

### Usage

#### To run the pipeline:

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Reproduce the pipeline:

   ```bash
   dvc repro
   # or
   python main.py
   ```

#### To run the web app:

1. Start the Flask app:

   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://localhost:8080` to use the web app for inference.

#### To dockerize the project:

1. Build the Docker image:

   ```bash
   docker build -t kidney_disease_classifier .
   ```
2. Run the Docker container:

   ```bash
   docker run -p 8080:8080 kidney_disease_classifier
   ```

### References

CT KIDNEY DATASET: https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone
