# Improved YOLOv13 with DTFL — Flower Classification on Embedded Devices

Official code for the paper:

> **Yuan, P., Yang, P., & Cui, W. (2026).** Improved YOLOv13 with dynamic threshold focal loss:
> Fine-grained flower classification on embedded devices in precision horticulture.
> *Smart Agricultural Technology*, **14**, 102112.
> [https://doi.org/10.1016/j.atech.2026.102112](https://doi.org/10.1016/j.atech.2026.102112)

Archived release: [10.5281/zenodo.17721837](https://doi.org/10.5281/zenodo.17721837)

---

## Overview

We restructure the detection-oriented YOLOv12 / YOLOv13 frameworks into **classification-specific
architectures** and introduce **Dynamic Threshold Focal Loss (DTFL)**, extending the ATFL
mechanism from the detection domain to classification and adapting it to multi-scale YOLO models
(N/S/M/L/E).

Key results:

| Dataset | Model | Top-1 | Params | Size | Speed |
|---|---|---|---|---|---|
| Private 100-class flowers | YOLOv13n+DTFL | **92.20 %** | 1.4 M | 5.76 MB | 19–20 FPS |
| Oxford 102 Flowers | YOLOv13n+DTFL | **98.04 %** | 1.4 M | ~5.8 MB | 19–20 FPS |
| Oxford 17 Flowers | YOLOv13n+DTFL | **98.16 %** | 1.4 M | ~8.0 MB | 19–20 FPS |

Without specialised training strategies, this matches or exceeds SOTA lightweight models while
remaining deployable on resource-constrained edge hardware.

---

## Repository contents

```
.
├── LICENSE                  MIT
├── NOTICE                   upstream attribution (YOLOv13 / Ultralytics / datasets)
├── README.md                this file
├── README_android.md        Android deployment guide (see below)
├── dtfl.py                  Dynamic Threshold Focal Loss implementation
├── requirements.txt
├── models/                  model configuration YAMLs (YOLOv13-cls, YOLOv12-cls)
├── yolov13n.pt              base checkpoint
└── android_app/             on-device Android application (APK source)
```

### `android_app/` — on-device inference

A complete Android application that runs the exported ONNX weights **fully offline** on the
phone: camera or gallery → local ONNX Runtime inference → Top-5 species with confidence.

- Three bundled models: private 100-class, Oxford 17, Oxford 102
- **Bilingual UI** (Chinese / English) following the system language, with Latin scientific
  names in English mode
- On-device results are numerically equivalent to PyTorch (max |Δ| ≈ 4e-7, Top-1 agreement
  100/100, 17/17, 102/102)
- Requires Android 8.0+ on a 64-bit ARM device

See [`android_app/README.en.md`](android_app/README.en.md) for the full English documentation
(build requirements, device requirements, metadata format, label-mapping pitfalls) and
[`android_app/README.md`](android_app/README.md) for the Chinese version.

---

## Quick start (training / inference)

```bash
conda create -n yolov13 python=3.11
conda activate yolov13
pip install -r requirements.txt
pip install -e .

# Copy DynamicThresholdFocalLoss from dtfl.py into ultralytics/utils/loss.py,
# then replace the existing v8ClassificationLoss with the one from dtfl.py.

from ultralytics import YOLO

model = YOLO('models/yolov13n-cls.yaml')
model.load('yolov13n.pt')

results = model.train(
    data='path/to/images',
    epochs=400, batch=8, imgsz=640,
    scale=0.5,       # S/M/L/X: 0.9
    mosaic=1.0, mixup=0.0,
    copy_paste=0.1,  # S: 0.15; M: 0.4; L: 0.5; X: 0.6
    project='runs/train', name='exp',
)
```

`data` points to an ImageFolder-style directory (`images/train/<class>/...`).

---

## Upstream projects

The network architecture is **YOLOv13**, which is itself based on **Ultralytics YOLO**:

- YOLOv13 — https://github.com/iMoonLab/yolov13 (AGPL-3.0)
- Ultralytics — https://github.com/ultralytics/ultralytics (AGPL-3.0)

Neither project's source code is vendored in this repository; both are referenced by URL.
`dtfl.py` and the Android application source are original work. See [`NOTICE`](NOTICE) for
the full attribution and licensing discussion.

---

## Datasets

**No dataset images are distributed in this repository** — only trained model weights.

| Dataset | Source | Notes |
|---|---|---|
| Private 100-class flowers | 7th National Collegiate Computer Contest (2025) organisers | 19,927 images, 100 classes, naturally imbalanced. Copyright remains with the organisers. |
| Oxford 17 Flowers | Nilsback & Zisserman, University of Oxford | 1,360 images, 17 classes. |
| Oxford 102 Flowers | Nilsback & Zisserman, University of Oxford | 8,189 images, 102 classes. |

The Oxford datasets publish no formal licence; the images were acquired by web search and
photography. Please cite the original dataset papers:

```bibtex
@inproceedings{nilsback2008flowers,
  title     = {Automated flower classification over a large number of classes},
  author    = {Nilsback, Maria-Elena and Zisserman, Andrew},
  booktitle = {Sixth Indian Conference on Computer Vision, Graphics \& Image Processing},
  year      = {2008},
  doi       = {10.1109/ICVGIP.2008.47}
}
```

---

## Citation

```bibtex
@article{yuan2026dtfl,
  title   = {Improved YOLOv13 with dynamic threshold focal loss: Fine-grained flower
             classification on embedded devices in precision horticulture},
  author  = {Yuan, Peiyin and Yang, Peixin and Cui, Weifu},
  journal = {Smart Agricultural Technology},
  volume  = {14},
  pages   = {102112},
  year    = {2026},
  doi     = {10.1016/j.atech.2026.102112}
}
```

---

## Licence

MIT — see [`LICENSE`](LICENSE).
