# flower-classification-cea-

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

model = YOLO('.../yolov13-main/ultralytics/cfg/models/v13/yolo-cls.yaml')
model.load('.pt')
# Train the model
results = model.train(
  data='.../images',
  epochs=400,
  batch=8,
  imgsz=640,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  device='',
  project='runs/train',
  name='exp',
  resume=True,
)

When running a YOLO classification task, copy the DynamicThresholdFocalLoss function from dtfl.py into ...\ultralytics\utils\loss.py. Then, replace the existing v8ClassificationLoss function in loss.py with the v8ClassificationLoss function from dtfl.py. You can specify the model scale to run within the v8ClassificationLoss function.
Oxford102：https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
Oxford17：https://www.robots.ox.ac.uk/~vgg/data/flowers/17/
YOLOv13 download link：https://github.com/iMoonLab/yolov13
