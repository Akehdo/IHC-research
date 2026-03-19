# ResNet50 No Augmentation Baseline Report

## Overview

This report presents the baseline experiment for HER2 image classification using a ResNet50 architecture on the patch-based dataset without augmentation.

## Dataset

See:
- `results/tables/dataset_summary.md`

## Experiment Setup

See:
- `results/tables/resnet50/resnet50_no_augmentation_experiment_setup.md`

## Overall Performance

See:
- `results/tables/resnet50/resnet50_no_augmentation_overall_metrics.md`

## Per-Class Performance

See:
- `results/tables/resnet50/resnet50_no_augmentation_class_metrics.md`

## Confusion Matrix

See:
- `results/tables/resnet50/resnet50_no_augmentation_confusion_matrix.md`

## Preliminary Interpretation

- ResNet50 is used as the baseline CNN architecture without augmentation.
- The model achieved `0.95` accuracy and `0.94` macro F1-score on the test split.
- The strongest class is `3+`, while the main confusion remains between classes `0` and `1+`.

## Future Work

- Compare ResNet50 with other architectures such as ResNet18, DenseNet121, and EfficientNet-B0.
- Add comparison tables for multiple architectures.
- Investigate class imbalance handling for class `2+` if needed.
