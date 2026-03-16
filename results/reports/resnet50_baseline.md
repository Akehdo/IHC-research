# ResNet50 Baseline Report

## Overview

This report presents the baseline experiment for HER2 image classification using a ResNet50 architecture on the patch-based dataset.

## Dataset

See:
- `results/tables/dataset_summary.md`

## Experiment Setup

See:
- `results/tables/resnet50_experiment_setup.md`

## Overall Performance

See:
- `results/tables/resnet50_overall_metrics.md`

## Per-Class Performance

See:
- `results/tables/resnet50_class_metrics.md`

## Confusion Matrix

See:
- `results/tables/resnet50_confusion_matrix.md`

## Preliminary Interpretation

- ResNet50 is used as the baseline CNN architecture.
- The model performance should be analyzed both overall and per class.
- Special attention should be paid to class `2+`, since it has fewer samples and may be harder to classify.

## Future Work

- Compare ResNet50 with other architectures such as ResNet18, DenseNet121, and EfficientNet-B0.
- Add comparison tables for multiple architectures.
- Investigate class imbalance handling if needed.
