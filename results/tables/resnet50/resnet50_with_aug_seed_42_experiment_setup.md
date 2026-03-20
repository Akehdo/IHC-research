# ResNet50 With Aug Seed 42 Experiment Setup

| Parameter | Value |
|---|---|
| Model | ResNet50 |
| Input image size | 224 x 224 |
| Training setup | With augmentation |
| Train transform | Resize, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor |
| Test transform | Resize, ToTensor |
| Batch size | 32 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | CrossEntropyLoss |
| Number of classes | 4 |
| Train split path | `data/Patch-based-dataset/train_data_patch` |
| Test split path | `data/Patch-based-dataset/test_data_patch` |
| Seed | 42 |
| Saved weights path | `models/her2_resnet50_with_aug_seed_42.pth` |
