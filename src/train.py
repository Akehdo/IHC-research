from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "Patch-based-dataset"
TRAIN_DIR = DATASET_ROOT / "train_data_patch"
TEST_DIR = DATASET_ROOT / "test_data_patch"
MODEL_PATH = PROJECT_ROOT / "models" / "her2_resnet50.pth"


def main():

    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # TRANSFORMS
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # DATASET
    train_dataset = ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    test_dataset = ImageFolder(
        TEST_DIR,
        transform=test_transform
    )

    print("Classes:", train_dataset.classes)

    # DATALOADER
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # MODEL
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model = model.to(device)

    # LOSS + OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # TRAIN LOOP
    epochs = 10

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        loop = tqdm(train_loader)

        for images, labels in loop:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    # EVALUATION
    print("\nEvaluating model...")

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in tqdm(test_loader):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"\nTest Accuracy: {accuracy:.2f}%")

    # SAVE MODEL
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"Model saved as: {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
