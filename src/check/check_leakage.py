import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
train_dir = PROJECT_ROOT / "data" / "Patch-based-dataset" / "train_data_patch"
test_dir = PROJECT_ROOT / "data" / "Patch-based-dataset" / "test_data_patch"

train_files = set()
test_files = set()

# collect train filenames
for root, dirs, files in os.walk(train_dir):
    for f in files:
        train_files.add(f)

# collect test filenames
for root, dirs, files in os.walk(test_dir):
    for f in files:
        test_files.add(f)

intersection = train_files.intersection(test_files)

print("Train files:", len(train_files))
print("Test files:", len(test_files))
print("Common files:", len(intersection))

if len(intersection) > 0:
    print("Potential leakage examples:")
    print(list(intersection)[:10])
else:
    print("No filename overlap detected")
