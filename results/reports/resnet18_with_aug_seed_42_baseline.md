# ResNet18 With Aug Seed 42 Baseline Report

## Overview

This report presents the ResNet18 experiment for HER2 image classification on the patch-based dataset with augmentation and seed 42.

## Dataset

See:
- `results/tables/dataset_summary.md`

## Experiment Setup

See:
- `results/tables/resnet18/resnet18_with_aug_seed_42_experiment_setup.md`

## Overall Performance

See:
- `results/tables/resnet18/resnet18_with_aug_seed_42_overall_metrics.md`

## Per-Class Performance

See:
- `results/tables/resnet18/resnet18_with_aug_seed_42_class_metrics.md`

## Confusion Matrix

See:
- `results/tables/resnet18/resnet18_with_aug_seed_42_confusion_matrix.md`

## Preliminary Interpretation

- ResNet18 with augmentation achieved `0.94` accuracy and `0.93` macro F1-score on the test split.
- The strongest class is `3+`, while the main confusion remains between classes `0` and `1+`.
- Compared with a stronger ResNet50 baseline, this run can be used as a lighter architecture baseline for comparison.

## Future Work

- Compare ResNet18 and ResNet50 under the same augmentation setting.
- Try class imbalance handling for class `2+` if needed.
- Add a final model comparison table across architectures.
