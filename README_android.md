# Flower Classification on Android — YOLOv13 + DTFL (Edge Inference)

An Android application that performs **fully offline flower classification on the phone itself**.
Point the camera at a flower (or pick a photo from the gallery) and the app returns the Top-5
species with confidence scores. No network connection, no cloud inference — the ONNX model runs
locally through ONNX Runtime.

This app is the deployment artefact accompanying the paper:

> Yuan, P., Yang, P., & Cui, W. (2026). *Improved YOLOv13 with dynamic threshold focal loss:
> Fine-grained flower classification on embedded devices in precision horticulture.*
> **Smart Agricultural Technology**, 14, 102112.
> https://doi.org/10.1016/j.atech.2026.102112

---

## 1. What this is (and what it is not)

The underlying weights are **classification** models, not detectors:

```
task      = classify
output    = (1, N) probability distribution   ← no bounding boxes
imgsz     = 640
```

| You want | Status |
|---|---|
| "Which species is this flower?" (17 / 100 / 102 classes) | ✅ Supported |
| Draw a box around each flower, count them | ❌ Not supported |
| Multiple different flowers in one photo | ❌ One overall label per image |

If you need localisation or counting, train a **detection** weight
(YOLOv13-detect with bounding-box annotations) and adapt the output decoding in
`ClassifierEngine.kt` to parse `(1, 4+nc, 8400)` tensors.

---

## 2. Bundled models

Three models ship inside `app/src/main/assets/models/`:

| Directory | Dataset | Classes | ONNX size | Top-1 (paper) |
|---|---|---|---|---|
| `Flower100_YOLOv13s` | Private 100-class flower dataset | 100 | 17.1 MB | 92.2 % |
| `OxFlowers17_YOLOv13s` | Oxford 17 Flowers | 17 | 16.7 MB | 98.16 % |
| `OxFlowers102_YOLOv13s` | Oxford 102 Flowers | 102 | 17.1 MB | 98.04 % |

The default model is `Flower100_YOLOv13s`. Users can switch models at runtime from the app
("Switch model"), and can also import their own `.modelzip` / `.onnx` bundle.

### On-device verification

Measured by exporting the same weights to ONNX and comparing against the PyTorch reference
(`export/e2e_check.py`, one image per class):

| Model | Sampled accuracy | Max |Δ| vs PyTorch | Top-1 agreement |
|---|---|---|---|
| Flower100_YOLOv13s | **96.0 %** (96/100) | 3.576e-07 | **100/100** |
| OxFlowers17_YOLOv13s | **100.0 %** (17/17) | 4.172e-07 | **17/17** |
| OxFlowers102_YOLOv13s | **97.1 %** (99/102) | 1.609e-06 | **102/102** |

The "Top-1 agreement" column is the important one: it shows the on-device pipeline is
**numerically equivalent** to the PyTorch results reported in the paper — differences are at
FP32 rounding noise, not approximation error.

---

## 3. Requirements

### 3.1 Device requirements

| Item | Requirement | Notes |
|---|---|---|
| **Android version** | **8.0 (API 26) or newer** | `minSdk 26`. Older devices cannot install the APK. |
| **CPU architecture** | **arm64-v8a (64-bit ARM)** | The APK ships only `arm64-v8a` native libraries. 32-bit (`armeabi-v7a`) devices will install but crash on launch with `UnsatisfiedLinkError: libonnxruntime.so not found`. |
| **RAM** | ≥ 4 GB recommended | `largeHeap=true`; peak usage ≈ 300–500 MB with a 640×640 float tensor. |
| **Storage** | ≈ 80 MB for the APK, 100–120 MB installed | |
| **Camera** | Optional (`required=false`) | Without a camera the app still works via gallery / share-intent. |

Practically: any mainstream phone released from ~2017 onwards (Snapdragon 660+, Kirin 970+,
Dimensity series) satisfies all of the above.

### 3.2 Permissions

The app requests **only** `android.permission.CAMERA`, and it is optional:

- Capture goes through the system camera app + `FileProvider` — **no storage permission needed**.
- Gallery selection uses the system Photo Picker on Android 13+ and `ACTION_OPEN_DOCUMENT` on
  older versions — **also permission-free**.
- The app registers an `ACTION_SEND` / `image/*` intent filter, so you can long-press a photo in
  the gallery → Share → "Flower Identifier" and skip the in-app picker entirely.

### 3.3 Build requirements

| Tool | Version used |
|---|---|
| JDK | 17 |
| Android Gradle Plugin | 8.7.x |
| Gradle | 8.9 |
| Kotlin | 2.x |
| `compileSdk` | 36 (Android 16) |
| `targetSdk` | 35 (Android 15) |
| `minSdk` | 26 (Android 8.0) |
| Build tools | 36.1.0 |
| ONNX Runtime Android | `com.microsoft.onnxruntime:onnxruntime-android:1.20.0` |

`compileSdk` only affects which APIs are available at compile time; it does not raise the
runtime floor. `targetSdk 35` means the app is behaviour-tested against Android 15 and runs in
compatibility mode on newer releases.

### 3.4 Building from source

```bash
# 1. Point Gradle at your SDK (create local.properties, never commit it)
echo "sdk.dir=/path/to/Android/Sdk" > local.properties

# 2. Build
./gradlew assembleDebug

# 3. Output
#    app/build/outputs/apk/debug/app-debug.apk
```

The models in `app/src/main/assets/models/` are already committed, so a plain clone builds a
fully functional APK with no extra downloads.

---

## 4. Project layout

```
FlowerClassifier/
├── app/
│   ├── build.gradle.kts            # SDK levels, ABI filters, ONNX Runtime dependency
│   └── src/main/
│       ├── AndroidManifest.xml     # CAMERA (optional), FileProvider, ACTION_SEND
│       ├── assets/models/          # Three bundled ONNX models + metadata.json
│       ├── java/com/flowerdemo/yolov13/
│       │   ├── MainActivity.kt     # UI wiring, camera/gallery/share-intent handling
│       │   ├── data/
│       │   │   ├── ImageLoader.kt      # bounds-aware downsampling + EXIF rotation
│       │   │   └── ModelRepository.kt  # builtin scan, import, selection persistence
│       │   ├── ml/
│       │   │   ├── ClassifierEngine.kt # ONNX Runtime session, tensor flattening
│       │   │   ├── ImagePreprocessor.kt# short-side resize → center crop → CHW
│       │   │   └── ModelMeta.kt        # metadata.json parsing, bilingual labels
│       │   ├── ui/MainViewModel.kt     # state machine, model switching, inference
│       │   └── util/LocaleHelper.kt    # zh vs. Latin-scientific name selection
│       └── res/
│           ├── layout/             # activity_main, dialog_models, item_*
│           ├── values/strings.xml      # Chinese (default)
│           └── values-en/strings.xml   # English
└── README.md                       # this file
```

---

## 5. Bilingual UI

The app ships two complete string sets and follows the **system language**:

- Chinese system → Chinese UI, Chinese species names.
- Any other language → English UI, **Latin scientific names** for species.

The English species labels come from the `names_english` array in each `metadata.json`, which is
positionally aligned with the Chinese `names` array. If a metadata file lacks `names_english`, or
its length does not match, the app automatically falls back to the Chinese table (so indices can
never go out of range).

---

## 6. Preprocessing must match training exactly

This is the most common cause of "the model works on PC but not on the phone".

```
classify_transforms():   short side → imgsz  →  center crop  →  ÷255  →  CHW
```

Two details that are easy to get wrong:

- **`DEFAULT_CROP_FRACTION` in the ultralytics classify pipeline is `1.0`, not the `0.875`
  used by ImageNet eval.** So `scale_size == imgsz == 640`: no extra margin crop.
- **`DEFAULT_MEAN = (0,0,0)`, `DEFAULT_STD = (1,1,1)`** — normalisation degenerates to a plain
  `/255`. Do not apply ImageNet statistics.

`ImagePreprocessor.kt` implements exactly this pipeline, and `metadata.json` carries the same
parameters so a re-exported model stays consistent.

---

## 7. Label mapping — the traps we hit

Every one of the three datasets needed a *different* mapping, and getting it wrong produces a
model that looks accurate in an end-to-end check but shows the wrong species name on screen.

### 7.1 Ultralytics orders class indices by **directory-name string sort**

This is the critical trap. Class index *i* corresponds to the *i-th directory name in
lexicographic order*, not numeric order:

```python
# Private 100-class dataset
weight names (first 12): ['164','165','166','167','169','171','172','173','1734','174','1743','1747']
numeric order:           ['164','165','166','167','169','171','172','173','174','176','177','178']
                                                              ↑ diverges at idx 8  ('1734' < '174')

# Oxford 102
weight names (first 12): ['1','10','100','101','102','11','12','13','14','15','16','17']
numeric order:           ['1','2','3','4','5','6','7','8','9','10','11','12', ...]
                          ↑ diverges at idx 1  ('10' < '2')
```

Cause: `ImageFolder` assigns class indices by sorting directory names as strings. Generating
labels in numeric order silently shifts the entire table.

**Mitigation:** derive the label table from the weight's own `names` order
(`export/make_oxford_metadata.py`), never from a numeric sort.

### 7.2 Per-dataset mapping rules

| Dataset | Directory name means | Weight `names` order | Mapping used |
|---|---|---|---|
| Private 100 | `category_id` (164, 1734, …) | lexicographic | dir → CSV lookup → Chinese name |
| Oxford 17 | English flower name (`BlueBale`, …) | alphabetical (= dir order) | dir → bilingual table |
| Oxford 102 | **1-based class id** (1…102) | lexicographic | dir → `int()` → official list `[cid-1]` |

For Oxford 102 the official class-name list (`torchvision.datasets.Flowers102.classes`) is
**0-based**, while the directory names are the **1-based** `class_id`. Hence
`official_list[int(dirname) - 1]`.

### 7.3 Why an end-to-end accuracy check cannot validate the mapping

If ground truth and prediction both look up the *same* `names` table, a whole-table shift makes
both sides wrong in the same way — accuracy stays high. Label correctness must be verified
against an **independent external source**.

For Oxford 102 we validated three independent ways:

1. Official category page sample images → reverse-lookup `class_id` in the dataset CSV → compare
   species names: **102/102 match**.
2. Official per-class image counts vs. CSV statistics: **102/102 match** (independently proves
   the 1-based semantics).
3. Deliberately using the wrong offset (`[cid]` instead of `[cid-1]`): **0/102 match**.

---

## 8. Model metadata format

Each model directory contains `best.onnx` plus `metadata.json`:

```jsonc
{
  "app_schema": 1,
  "name": "OxFlowers102_YOLOv13s",
  "task": "classify",
  "format": "onnx",
  "model_file": "best.onnx",
  "imgsz": 640,
  "batch": 1, "channels": 3, "layout": "NCHW", "color_order": "RGB",
  "normalize":  { "scale": 0.00392156862745098, "mean": [0,0,0], "std": [1,1,1] },
  "preprocess": { "resize_mode": "short_side", "crop": "center",
                  "resize_size": 640, "crop_size": 640 },
  "output":     { "activation": "softmax" },
  "num_classes": 102,
  "names":         ["粉花报春花", "蓝刺头", ...],          // Chinese, index-aligned
  "names_english": ["pink primrose", "globe thistle", ...], // Latin/common, index-aligned
  "names_dir":     ["1","10","100", ...]                    // original weight order
}
```

`output.activation == "softmax"` tells the app the model head already applies softmax; otherwise
the app applies it itself.

---

## 9. Re-exporting a model

The export scripts live outside this directory (they need the training project's `ultralytics`
source tree, because the `.pt` checkpoints pickle `AdaptiveThresholdFocalLoss`).
See `export/export_for_android.py` and `export/make_oxford_metadata.py` in the parent project.

Two practical notes:

- The scripts accept `--project-root` so they pick up the *training* project's `ultralytics`
  rather than whichever one happens to be first on `sys.path`.
- `metadata.json` for classification models must be generated in **weight-`names` order**, not
  numeric order (see §7.1).

---

## 10. Datasets and attribution

**No dataset images are distributed in this repository** — only trained model weights. Model
weights do not contain original image pixels.

| Dataset | Source | Terms |
|---|---|---|
| Oxford 17 / 102 Flowers | Nilsback & Zisserman, Visual Geometry Group, University of Oxford | No formal licence is published; images were web-crawled and photographed. Cite the papers below. |
| Private 100-class dataset | Released by the organisers of the 7th National Collegiate Computer Contest (2025) | Copyright remains with the contest organisers. |

Please cite:

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

@inproceedings{nilsback2008flowers,
  title     = {Automated flower classification over a large number of classes},
  author    = {Nilsback, Maria-Elena and Zisserman, Andrew},
  booktitle = {Sixth Indian Conference on Computer Vision, Graphics \& Image Processing},
  year      = {2008},
  doi       = {10.1109/ICVGIP.2008.47}
}
```

---

## 11. Licence

MIT — see [`LICENSE`](../LICENSE).

**Upstream attribution.** The network architecture is YOLOv13
([`iMoonLab/yolov13`](https://github.com/iMoonLab/yolov13)), which is itself based on
[Ultralytics](https://github.com/ultralytics/ultralytics). Training code is **not** vendored in
this repository; the upstream projects are referenced by URL only. See
[`NOTICE`](../NOTICE) for details.

---

## 12. Android version FAQ

**Q: Which Android version do I need?**
Android 8.0 (API 26) or newer, on a 64-bit ARM device.

**Q: Why does it crash immediately on my older phone?**
The APK ships only `arm64-v8a` native libraries. A 32-bit device cannot load
`libonnxruntime.so`. Check your ABI with CPU-Z or AIDA64 — you need `arm64-v8a` in the list.

**Q: Will it install on Android 16?**
Yes. `targetSdk 35` is a behaviour target, not an upper bound; newer Android releases run
older-target apps in compatibility mode.

**Q: Can I install this alongside other apps?**
Yes. The package name is `com.flowerdemo.yolov13`, which does not collide with anything else in
this project.

**Q: How do I get 32-bit support?**
You would need to add `armeabi-v7a` to `abiFilters` and obtain 32-bit ONNX Runtime native
libraries (ORT 1.20 no longer ships them; you would have to fall back to ≤ 1.16 or build a
custom CPU-only package), and lower `minSdk` with backward-compatible branches for
`FileProvider` and `PickVisualMedia`. Expect 1–2 s per inference on such hardware.
