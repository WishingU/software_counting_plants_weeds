# Broadleaf Weed Detection and Counting Model Development Report

## 1. Project Overview

This project aims to develop an object detection model for automatically detecting and counting broadleaf weeds in images. The model outputs the location, class, and confidence score of each detected plant and estimates the total number of plants in an image.

The target application involves relatively sparse scenes containing plants of different sizes.

## 2. Initial Model

### 2.1 Model Description

The initial model was based on YOLO26m and configured as a single-class object detection model. The target class was `broadleaf_weed`.

| Parameter | Value |
| --- | --- |
| Model | YOLO26m |
| Task | Single-class object detection |
| Class | `broadleaf_weed` |
| Input size | 1024 (`imgsz`, aspect ratio preserved with padding) |
| Training epochs | 92 |
| Evaluation metrics | Precision, Recall, mAP50, mAP50-95, Count MAE |

### 2.2 Initial Model Limitations

The initial model provided basic broadleaf weed detection capability, but its performance on local images was not sufficiently stable. Differences in soil appearance, lighting, camera angle, and plant appearance caused missed detections and inaccurate counting.

Further fine-tuning was therefore required to adapt the model to the target scene.

## 3. First Fine-Tuning Iteration

### 3.1 Objective

The first fine-tuning stage focused on adapting the model to the local image environment and improving detection and counting performance.

### 3.2 Training Approach

The YOLO26m model was fine-tuned using locally annotated images while maintaining the single-class `broadleaf_weed` detection task. A local validation set was used for model selection, followed by evaluation on a local test set.

### 3.3 Results

The fine-tuned model showed clear improvements on local images:

- Higher detection recall;
- Improved mAP50;
- Lower counting error;
- Better adaptation to local soil, pot, and plant appearance.

However, the amount of local training data was limited. As a result, the model became more dependent on the specific backgrounds and visual patterns in the local images. This created a risk of overfitting and motivated the use of mixed-dataset fine-tuning.

## 4. Mixed-Dataset Fine-Tuning

### 4.1 Training Strategy

The second training stage combined a representative subset of the external training data with local target-scene images.

The external data was used as the main training source, while the local images were added for target-scene adaptation. Local images were sampled more frequently, and a lower learning rate was used to reduce excessive changes to the pretrained model.

This strategy was designed to preserve general plant features while improving adaptation to the local sparse-scene environment.

### 4.2 Training Parameters

| Parameter | Value |
| --- | ---: |
| Model | YOLO26m |
| Input size | 768 (`imgsz`, aspect ratio preserved with padding) |
| Training epochs | 6 |
| Batch size | 2 |
| Optimizer | AdamW |
| Initial learning rate | 0.0005 |
| Weight decay | 0.0005 |
| Warmup epochs | 1 |

The best checkpoint was selected according to validation performance. The confidence threshold was selected using counting results on the validation set.

Recommended inference settings:

- Input size: 768;
- Confidence threshold: 0.40;
- IoU threshold: 0.70.

## 5. Final Model Evaluation

The model was evaluated using Precision, Recall, mAP50, mAP50-95, Count MAE, and Count bias. Count MAE was treated as the primary metric for practical counting performance.

### 5.1 Local Target Test Results

| Metric | Result |
| --- | ---: |
| Precision | 0.739 |
| Recall | 0.763 |
| mAP50 | 0.780 |
| mAP50-95 | 0.354 |
| Count MAE | 1.57 |
| Count bias | +0.14 |

The results indicate that the final model can detect broadleaf weeds relatively consistently in local sparse-scene images. The count bias is close to zero, suggesting that the model does not have a strong overall tendency to over-count or under-count.

### 5.2 Model Version Comparison

| Model Version | Main Characteristic |
| --- | --- |
| Initial pretrained model | Basic detection capability but insufficient adaptation to the local scene |
| Local-only fine-tuned model | Improved local performance but higher overfitting risk |
| Mixed fine-tuned model | Better balance between target-scene adaptation and general visual features |

## 6. Limitations

The current model has several limitations:

- The amount of locally annotated data is still limited;
- The local test set is relatively small;
- The model currently supports only one broadleaf weed class;
- Performance under significantly different lighting, camera angles, and plant growth stages requires further validation;
- Detection of heavily occluded, very small, or low-contrast plants can still be improved.

## 7. Future Work

Future development may focus on:

- Collecting local images from different dates and environments;
- Adding plants at different sizes and growth stages;
- Adding negative images containing empty soil or empty pots;
- Building a larger independent test set;
- Comparing input sizes of 640, 768, and 1024;
- Improving detection of small, occluded, and low-contrast plants.

## 8. Conclusion

The initial YOLO26m model provided a useful foundation for broadleaf weed detection, but its adaptation to the local image environment was insufficient. Fine-tuning with only a small local dataset improved local performance but introduced a risk of overfitting.

By using external data as the primary training source and adding local images for target-scene adaptation, the final mixed fine-tuned model achieved improved local detection and counting performance. The model is suitable for relatively sparse scenes containing plants of different sizes. Further improvements will depend mainly on collecting more diverse and independently captured local training images.
