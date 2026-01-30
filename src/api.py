import io

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import models, transforms

from src.config import API_PORT, MODEL_PATH


# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(title="Image Classification API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load model at startup
# -----------------------------
model = None
class_names = None


def load_model():
    global model, class_names

    # Validation transforms (same as training/evaluation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Load class names from validation dataset
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder("data/val")
    class_names = dataset.classes

    # Load model architecture
    num_classes = len(class_names)
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return transform


# Load preprocessing transform and model
transform = load_model()


# -----------------------------
# Health check endpoint
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Read file bytes
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid image file")

    # Preprocess image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item()

    return JSONResponse(
        status_code=200,
        content={
            "predicted_class": predicted_class,
            "confidence": confidence_score,
        },
    )
