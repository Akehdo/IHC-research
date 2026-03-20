# ResNet50 With Aug Seed 42 Baseline Report

## Overview

This report presents the baseline experiment for HER2 image classification using a ResNet50 architecture on the patch-based dataset with augmentation and seed 42.

## Dataset

See:
- `results/tables/dataset_summary.md`

## Experiment Setup

See:
- `results/tables/resnet50/resnet50_with_aug_seed_42_experiment_setup.md`

## Overall Performance

See:
- `results/tables/resnet50/resnet50_with_aug_seed_42_overall_metrics.md`

## Per-Class Performance

See:
- `results/tables/resnet50/resnet50_with_aug_seed_42_class_metrics.md`

## Confusion Matrix

See:
- `results/tables/resnet50/resnet50_with_aug_seed_42_confusion_matrix.md`

## Preliminary Interpretation

- ResNet50 is used as the baseline CNN architecture with augmentation.
- The model achieved `0.95` accuracy and `0.94` macro F1-score on the test split.
- The strongest class is `3+`, while the main confusion remains between classes `0` and `1+`.

## Future Work

- Compare ResNet50 with other architectures such as ResNet18, DenseNet121, and EfficientNet-B0.
- Add comparison tables for multiple architectures.
- Investigate class imbalance handling for class `2+` if needed.
