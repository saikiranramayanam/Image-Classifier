import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from src.config import MODEL_PATH


def train():
    # -----------------------------
    # Device configuration
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Data transforms
    # -----------------------------
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # -----------------------------
    # Datasets and loaders
    # -----------------------------
    train_dataset = datasets.ImageFolder("data/train", transform=train_transforms)
    val_dataset = datasets.ImageFolder("data/val", transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")

    # -----------------------------
    # Load pre-trained model
    # -----------------------------
    model = models.resnet50(pretrained=True)

    # Freeze convolutional layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final classification layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)

    # -----------------------------
    # Loss and optimizer
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    # -----------------------------
    # Training loop
    # -----------------------------
    best_val_acc = 0.0

    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}/5")

        # Training phase
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Training loss: {avg_loss:.4f}")

        # Validation phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f"Validation accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(os.path.dirname(MODEL_PATH)).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print("Best model saved.")

    print("Training completed.")


if __name__ == "__main__":
    train()
