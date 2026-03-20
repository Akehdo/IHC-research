# ResNet18 With Aug Seed 42 Experiment Setup

| Parameter | Value |
|---|---|
| Model | ResNet18 |
| Training setup | With augmentation |
| Input image size | 224 x 224 |
| Train transform | Resize, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor |
| Test transform | Resize, ToTensor |
| Batch size | 32 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | CrossEntropyLoss |
| Number of classes | 4 |
| Seed | 42 |
| Train split path | `data/Patch-based-dataset/train_data_patch` |
| Test split path | `data/Patch-based-dataset/test_data_patch` |
| Saved weights path | `models/her2_resnet18_with_aug_seed_42.pth` |
