import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from common import MODELS_DIR, TRAIN_DIR, build_run_name, get_device, get_model, get_transforms, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HER2 classifier")
    parser.add_argument("--model", choices=["resnet18", "resnet50"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    pin_memory = device.type == "cuda"

    train_transform, _ = get_transforms(use_augmentation=True)

    train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    model = get_model(args.model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    patience_counter = 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = build_run_name(args.model, args.seed, with_augmentation=True)
    model_path = MODELS_DIR / f"{run_name}.pth"


    for epoch in range(args.epochs):
        model.train()  # Switch model to training mode
        running_loss = 0.0  # Store total loss for this epoch

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images = images.to(device)  # Move input images to CPU or GPU
            labels = labels.to(device)  # Move true labels to CPU or GPU

            optimizer.zero_grad()  # Clear old gradients from previous step

            outputs = model(images)  # Forward pass: model predicts class scores
            loss = criterion(outputs, labels)  # Compute error between predictions and true labels

            loss.backward()  # Backward pass: compute gradients
            optimizer.step()  # Update model weights using gradients

            running_loss += loss.item()  # Add current batch loss to total epoch loss

        epoch_loss = running_loss / len(train_loader)  # Compute average loss for the epoch
        print(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to: {model_path}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")

        if patience_counter >= args.patience:
            print("Early stopping triggered")
            break


if __name__ == "__main__":
    main()
