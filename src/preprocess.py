import random
import shutil
from pathlib import Path

from torchvision.datasets import Caltech101


def preprocess_caltech101(
    output_dir="data",
    train_split=0.8,
    min_classes=10
):
    """
    Downloads Caltech-101, splits into train/val, and organizes
    images into ImageFolder-compatible structure.
    """

    # Output directories
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset (this creates the raw folder)
    dataset = Caltech101(root="caltech101_raw", download=True)

    # Raw images live here
    raw_image_root = Path(dataset.root) / "101_ObjectCategories"

    # Select at least 10 classes
    class_names = sorted([
        d.name for d in raw_image_root.iterdir()
        if d.is_dir()
    ])[:min_classes]

    print(f"Using {len(class_names)} classes:")
    for cls in class_names:
        print(f" - {cls}")

    for class_name in class_names:
        class_dir = raw_image_root / class_name
        images = list(class_dir.glob("*.jpg"))

        random.shuffle(images)
        split_idx = int(len(images) * train_split)

        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create class folders
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)

        # Copy train images
        for img_path in train_images:
            shutil.copy(img_path, train_dir / class_name / img_path.name)

        # Copy val images
        for img_path in val_images:
            shutil.copy(img_path, val_dir / class_name / img_path.name)

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess_caltech101()
