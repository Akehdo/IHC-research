# HER2 IHC Research

This repository contains a small PyTorch-based pipeline for HER2 immunohistochemistry image classification on a patch-based dataset.

## Project contents

- `src/train.py` trains a `ResNet50` classifier on `data/Patch-based-dataset/train_data_patch`.
- `src/evaluate.py` evaluates the saved model on `data/Patch-based-dataset/test_data_patch`.
- `src/check_duplicates.py` checks whether identical image files appear in both train and test splits.
- `src/check_leakage.py` checks overlap in filenames between train and test splits.
- `src/check_slide_leakage.py` checks overlap in slide identifiers between train and test splits.
- `notebooks/her2_classification.ipynb` contains the notebook version of the work.
- `models/her2_resnet50.pth` is the saved model weights.

## Dataset structure

The code expects the following folder structure:

```text
data/
`-- Patch-based-dataset/
    |-- train_data_patch/
    |   |-- class_0/
    |   |-- class_1+/
    |   |-- class_2+/
    |   `-- class_3+/
    `-- test_data_patch/
        |-- class_0/
        |-- class_1+/
        |-- class_2+/
        `-- class_3+/
```

Current class counts in this local copy:

### Train split

- `class_0`: 3031
- `class_1+`: 2151
- `class_2+`: 905
- `class_3+`: 2710

### Test split

- `class_0`: 758
- `class_1+`: 538
- `class_2+`: 226
- `class_3+`: 678

## Setup

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## How to run

Train the model:

```powershell
python src/train.py
```

Evaluate the saved model:

```powershell
python src/evaluate.py
```

Check data leakage:

```powershell
python src/check_duplicates.py
python src/check_leakage.py
python src/check_slide_leakage.py
```
## Project structure

```text
IHC-research/
|-- data/
|-- models/
|-- notebooks/
|-- src/
|-- README.md
|-- requirements.txt
`-- .gitignore
```
