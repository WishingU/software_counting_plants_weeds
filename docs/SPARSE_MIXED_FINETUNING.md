# Sparse-Scene Mixed Fine-Tuning

## Objective

The deployment target is broadleaf plant counting in relatively sparse scenes. Plants may be small or large. The training mixture therefore uses the diverse CropAndWeed broadleaf dataset as the primary source and the local 21 August photographs as a target-domain adaptation set.

## Dataset mixture

- Primary source: `data/cropandweed_broadleaf`.
- Target source: `data/21_08_borad_jpg`.
- The source checkpoint was already pretrained on all 6,367 external training images.
- Fine-tuning external replay subset: 2,000 deterministically selected images.
- Target training images: 42.
- Target replay factor: 16.
- Effective target training images: 672.
- Effective fine-tuning target image share: approximately 25.1%; external replay remains the 74.9% majority.
- Validation and test sets contain only the untouched target validation and test images.

The generated dataset uses hard links for images whenever possible, so the mixture does not duplicate the external image bytes. Labels are copied under unique names. The external data remains the clear majority while target replay is strong enough to influence optimization.

## Training decision

- Starting checkpoint: the CropAndWeed 1024 pretraining best checkpoint.
- Input size: 768 pixels.
- Full-model fine-tuning; no layers are frozen.
- Low initial learning rate: 0.0005.
- AdamW optimizer.
- Target-domain validation controls checkpoint selection and early stopping.
- Final evaluation must be reported separately on the target test split and the external CropAndWeed test split.

This setup is intended to reduce catastrophic forgetting while adapting to the local camera, soil, and sparse planting layout.

## Completed run

- Run: `runs/detect/yolo26m_broadleaf_sparse_mixed_768_v2`.
- Training: six epochs, stopped normally after 0.569 hours.
- Best checkpoint selected at epoch 3 by Ultralytics fitness.
- Best target validation: precision 0.650, recall 0.828, mAP50 0.765, mAP50-95 0.268.
- Count threshold selected on validation: confidence 0.40 with IoU 0.70.

## Final evaluation

| Split | Images | Precision | Recall | mAP50 | mAP50-95 | Count MAE | Count bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Local target test | 7 | 0.739 | 0.763 | 0.780 | 0.354 | 1.57 | +0.14 |
| CropAndWeed external test | 995 | 0.633 | 0.562 | 0.607 | 0.383 | 1.94 | -1.37 |

The mixed model retains far more external capability than the earlier target-only fine-tune, whose external mAP50 fell to approximately 0.249. The original external checkpoint remains slightly better on its own test set (mAP50 approximately 0.646), while the mixed model is substantially better on the local target domain.

The application defaults for this model are 768 pixel inference, confidence 0.40, and IoU 0.70.
