import json
import os
from pathlib import Path

import torch
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from src.config import MODEL_PATH


def evaluate():
    # -----------------------------
    # Device configuration
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Validation transforms (NO augmentation)
    # -----------------------------
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # -----------------------------
    # Validation dataset & loader
    # -----------------------------
    val_dataset = datasets.ImageFolder("data/val", transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )

    num_classes = len(val_dataset.classes)
    print(f"Evaluating on {num_classes} classes")

    # -----------------------------
    # Load model architecture
    # -----------------------------
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # -----------------------------
    # Inference loop
    # -----------------------------
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # -----------------------------
    # Metrics computation
    # -----------------------------
    accuracy = accuracy_score(all_labels, all_preds)
    precision_weighted = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    conf_matrix = confusion_matrix(all_labels, all_preds).tolist()

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "confusion_matrix": conf_matrix,
    }

    # -----------------------------
    # Save metrics to JSON
    # -----------------------------
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    evaluate()
