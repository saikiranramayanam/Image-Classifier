# 🖼️ End-to-End Image Classification System (Transfer Learning)

This project implements an **end-to-end image classification pipeline** using transfer learning with a pre-trained **ResNet50** model. It covers the complete ML workflow from dataset preparation to production deployment.

## 🎯 Objectives
- Apply transfer learning to Caltech-101 dataset
- Build reproducible training & evaluation pipeline
- Deploy model as containerized REST API
- Containerize with Docker & orchestrate with Docker Compose
- Run entire system with **single command**

## 📊 Dataset
- **Dataset**: Caltech-101 (multi-class image classification)
- **Split**: 80% train / 20% validation
- **Preprocessing**: Automatic download + ImageFolder structure
- **Classes**: At least 10 (configurable)

## 🧠 Model & Training

| Feature | Details |
|---------|---------|
| **Architecture** | ResNet50 (ImageNet pre-trained) |
| **Framework** | PyTorch |
| **Strategy** | Freeze conv layers, train final classifier |
| **Augmentation** | RandomHorizontalFlip, RandomRotation |
| **Loss** | CrossEntropyLoss |
| **Optimizer** | Adam |
| **Output** | `model/image_classifier.pth` (best val accuracy) |

## 📈 Evaluation Metrics

Evaluation on validation set saves to `results/metrics.json`:

```json
{
  "accuracy": 0.68,
  "precision_weighted": 0.70,
  "recall_weighted": 0.68,
  "confusion_matrix": [[...]]
}
🌐 REST API Endpoints
✅ Health Check
GET /health
Response:

json
{ "status": "ok" }
🖼️ Image Prediction
POST /predict
Content-Type: multipart/form-data
Form field: file (image)
Success Response:

json
{
  "predicted_class": "airplanes",
  "confidence": 0.99
}
Error Handling: 400/422 for missing/invalid files

🗂️ Project Structure
image-classifier/
├── data/                    # Preprocessed dataset
│   ├── train/
│   └── val/
├── model/                   # Trained model
│   └── image_classifier.pth
├── results/                 # Evaluation outputs
│   └── metrics.json
├── src/
│   ├── config.py           # Environment config
│   ├── preprocess.py       # Dataset prep
│   ├── train.py            # Training
│   ├── evaluate.py         # Evaluation
│   └── api.py              # FastAPI server
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
🚀 Quick Start
1. Setup Environment
bash
cp .env.example .env
# Edit .env:
# API_PORT=8000
# MODEL_PATH=model/image_classifier.pth
⚠️ Never commit .env to version control.

2. Run Full Pipeline
bash
# Train, evaluate, and start API
docker-compose up --build
3. Test API
bash
# Health check
curl http://localhost:8000/health

# Predict (after training completes)
curl -X POST http://localhost:8000/predict \
  -F "file=@data/val/airplanes/image_0001.jpg"
🐳 Containerization
Base Image: Python 3.11 slim

Model: Mounted as volume (no retraining in container)

Health Check: /health endpoint

Single Command: docker-compose up --build

📦 Dependencies
torch torchvision
fastapi uvicorn
pillow scikit-learn
python-multipart
🧪 Reproducibility & Best Practices
✅ Environment variables for config

✅ Clear separation: preprocess/train/eval/inference

✅ No secrets in repo

✅ Deterministic inference

✅ Health-checked containers

✅ Evaluation Readiness
Designed to pass:

Automated API tests

Docker & Compose validation

Code quality checks

Manual review

Author: Sai Kiran Ramayanam
Dataset: Caltech-101
Tech Stack: PyTorch, FastAPI, Docker