import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
train_dir = PROJECT_ROOT / "data" / "Patch-based-dataset" / "train_data_patch"
test_dir = PROJECT_ROOT / "data" / "Patch-based-dataset" / "test_data_patch"


def extract_slide(filename):
    return filename.split("_")[0]


train_slides = set()
test_slides = set()

for root, dirs, files in os.walk(train_dir):
    for f in files:
        slide = extract_slide(f)
        train_slides.add(slide)

for root, dirs, files in os.walk(test_dir):
    for f in files:
        slide = extract_slide(f)
        test_slides.add(slide)

intersection = train_slides.intersection(test_slides)

print("Train slides:", len(train_slides))
print("Test slides:", len(test_slides))
print("Common slides:", len(intersection))

if intersection:
    print("Leakage slides:", list(intersection)[:10])
else:
    print("No slide overlap detected")
