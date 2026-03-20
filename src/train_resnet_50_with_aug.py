from pathlib import Path
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm



PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "Patch-based-dataset"
TRAIN_DIR = DATASET_ROOT / "train_data_patch"
TEST_DIR = DATASET_ROOT / "test_data_patch"
MODEL_PATH = PROJECT_ROOT / "models" / "her2_resnet50_aug_seed_42.pth"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
SEED = 42
NUM_CLASSES = 4



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed: {SEED}")

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"TRAIN_DIR exists: {TRAIN_DIR.exists()}")
    print(f"TEST_DIR exists: {TEST_DIR.exists()}")


    # Transforms
    print("Creating transforms...")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("Transforms created successfully")


    # Dataset
    print("Loading train dataset...")
    train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)

    print("Loading test dataset...")
    test_dataset = ImageFolder(TEST_DIR, transform=test_transform)

    print("Datasets loaded successfully")
    print(f"Classes: {train_dataset.classes}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")


    # Dataloader
    print("Creating loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    print("Loaders created successfully")

    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"First batch labels: {labels[:10]}")


    # Model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    print("Model created successfully")
    print(model.fc)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Loss and optimizer created successfully")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()