# YOLOv10n DMS Control Experiment

This repository contains the YOLOv10n control experiment and dataset configuration for a Driver Monitoring System (DMS).

> Important: This repository was previously named `yolo-elder-fall-detection`. That name was inaccurate. The dataset and trained weights in this repository are for driver monitoring, not fall detection.

## Model

- Architecture: YOLOv10n
- Input size: 640 x 640
- Task: object detection
- Application: unsafe driving behavior detection

## Dataset Classes

1. `eyeclose`
2. `eyeopen`
3. `face`
4. `phone`
5. `yawn`

The extended six-class DMS model, including `smoke`, is maintained in [yolo-dms-smoke](https://github.com/tel0955850213/yolo-dms-smoke).

## Key Files

- `train_v10.py`: YOLOv10n training script
- `data.yaml`: five-class DMS dataset configuration
- `DMS_Thesis/YOLOv10_Control2/args.yaml`: recorded training configuration
- `best.pt`: trained YOLOv10n weights

## Related Fall-Detection Project

The robotic-dog fall-detection system uses a separate custom YOLO11s model and is maintained in [app_rescue_dog](https://github.com/tel0955850213/app_rescue_dog).
