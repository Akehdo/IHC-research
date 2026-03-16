import hashlib
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
train_dir = PROJECT_ROOT / "data" / "Patch-based-dataset" / "train_data_patch"
test_dir = PROJECT_ROOT / "data" / "Patch-based-dataset" / "test_data_patch"


def image_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


train_hashes = {}

print("Scanning train...")

for root, dirs, files in os.walk(train_dir):
    for file in files:
        path = os.path.join(root, file)
        h = image_hash(path)
        train_hashes[h] = path

print("Scanning test...")

duplicates = []

for root, dirs, files in os.walk(test_dir):
    for file in files:
        path = os.path.join(root, file)
        h = image_hash(path)

        if h in train_hashes:
            duplicates.append((train_hashes[h], path))

print("Total duplicates:", len(duplicates))

for d in duplicates[:10]:
    print("Train:", d[0])
    print("Test :", d[1])
    print()
