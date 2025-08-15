# OcuScan - Retinal Fundus Classification

Comprehensive pipeline for fundus-image classification with a consistent **80/10/10 stratified split**, a robust **preprocessing** stack, mixed-precision **training**, and a **Streamlit GUI** for inference.  

---

## 1) Overview

- Robust fundus preprocessing (ROI crop → shade correction → color constancy → CLAHE (G & L) → adaptive gamma → unsharp → letterbox+resize).
- Model zoo across **224/240/299/384** input sizes.
- **AMP** everywhere, checkpoint on **best val accuracy**, reproducible splits (`SEED=42`).
- Reports: **classification report**, **confusion matrix**, **ROC curves**, **AUC (macro/weighted/micro)**, **train/val curves**.

---

## 2) Data & Preprocessing
The dataset has 4,728 fundus images across five classes. Most samples are diabetic_retinopathy (23.2%, 1,098), followed by normal (22.7%, 1,074), cataract (22.0%, 1,038), and glaucoma (21.3%, 1,007). ARMD is the smallest class at 10.8% (511). Overall it’s fairly balanced among four classes, with ARMD under-represented (largest:smallest ≈ 2.15×), so class-weighted loss or macro metrics are a good idea.

<p align="center">
  <img src="images/download(1).png" alt="Pie Chart of Class Distribution" width="480">
</p>


### Core preprocessing steps (fundus-specific)
All steps occur in **`preprocess_fundus(img_bgr, size)`** in this order:

1. **ROI crop (`fundus_roi_crop`)**  
   - Threshold low intensities (`gray > 8`) to get a circular mask  
   - Morphological close (7×7) to fill holes  
   - **`robust_bbox_from_mask`**: bounding box with small margin → crop tightly to fundus

2. **Illumination/shade correction (`shade_correction`)**  
   - Divide by heavy Gaussian blur (σ≈40) to remove vignetting/shading; rescale

3. **Color constancy (`shades_of_gray_cc`, p=6)**  
   - Normalizes per-channel gains to reduce color cast

4. **Local contrast on green (`clahe_on_green`)**  
   - CLAHE on the **green channel** (vessels/lesions are prominent in G)

5. **Optional global contrast (`optional_l_channel_clahe`)**  
   - CLAHE on **L (lightness)** in LAB space for overall contrast

6. **Adaptive gamma (`adaptive_gamma`, target=0.42)**  
   - Computes image median brightness → sets gamma to reach target midtone

7. **Sharpen (`unsharp`, σ=1.0, amount=0.5)**  
   - Unsharp masking for detail enhancement

8. **Square letterbox + resize (`letterbox_square`)**  
   - Pads to square (no aspect distortion) and resizes to **224×224 or any other size** (cubic)

> Each step returns `uint8` BGR for OpenCV.

---

## Parallel processing
- Collects all image files via `rglob`
- Uses `ThreadPoolExecutor(workers)` to **process images concurrently**
- Per-file function: **`process_one`**
  - Skips if output already exists
  - Reads → `preprocess_fundus` → writes → returns status (`ok`, `skipped`, `read_error`, `write_error`, or `error:<msg>`)

---


**Tunable knobs:** CLAHE (clip/tile), shade σ, gamma target, unsharp amount, output size (224/240/299/384).

---

## 3) Model Zoo (current)

All use the same split protocol and training loop style (Adam/AdamW, ReduceLROnPlateau on **val acc**, AMP).

### 224×224 (torchvision)
- **ResNet-50** — `models.resnet50(weights=DEFAULT)` → replace `fc`
- **ResNet-101** — `models.resnet101(weights=DEFAULT)` → replace `fc`
- **MobileNetV3-Large** — replace `classifier[-1]`
- **DenseNet-121/169/201** — replace `classifier`

### 240×240
- **EfficientNet-B1** — resize to **(240,240)**; replace `classifier[1]`

### 299×299
- **InceptionV3** — aux logits (loss = CE(main) + **0.4**·CE(aux))
- **Inception-ResNet-v2 (timm)**

### 384×384
- **ConvNeXt (Tiny/Small/Base)** — optional gradient accumulation for VRAM

> **Removed:** Swin 224→384 two-stage.

---

## 4) Training Recipes

- **Optimizers**
  - CNNs/ResNet/EfficientNet/Inception: **Adam(lr=1e-4)**
  - ConvNeXt: **AdamW(lr=1e-4, weight_decay=5e-4)**
- **Schedulers**: `ReduceLROnPlateau(..., mode="max", factor=0.5, patience=2)` on **val acc**
- **Loss**: CrossEntropy (ViT/ConvNeXt previously used LS=0.1; not used now)
- **AMP**: `torch.amp.autocast` + `GradScaler`
- **Checkpoints**: `checkpoints/<model>_best.pth` with `model_state`, `epoch`, `val_acc`, `classes`

**Input roots**
- 224: `preprocessed224_best`
- 240 (B1): resize in transform
- 299 (Inception*): `preprocessed299_inception` or resize
- 384 (ConvNeXt): `preprocessed384_best`

---

## 5) Evaluation

Per model:
- **Classification report** (precision/recall/F1 per class)
- **Confusion matrix**
- **AUC (OVR)**: macro / weighted / micro (safe for missing classes)
- **ROC curves** (per class)
- **Train/Val curves** (acc, loss)

---

## 6) Results (fill in)

| Model | Input | Epochs | Optim | Val Acc (best) | Test Acc | Macro AUC | Weighted AUC | Micro AUC | Notes | Checkpoint |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| **ResNet-50** | 224 | 12 | Adam |  |  |  |  |  |  | `checkpoints/resnet50_best.pth` |
| **ResNet-101** | 224 | 12 | Adam |  |  |  |  |  |  | `checkpoints/resnet101_best.pth` |
| MobileNetV3-L | 224 | 10 | Adam |  |  |  |  |  |  | `checkpoints/mobilenetv3_best.pth` |
| DenseNet-121 | 224 | 12 | Adam |  |  |  |  |  |  | `checkpoints/densenet121_best.pth` |
| DenseNet-169 | 224 | 12 | Adam |  |  |  |  |  |  | `checkpoints/densenet169_best.pth` |
| DenseNet-201 | 224 | 12 | Adam |  |  |  |  |  |  | `checkpoints/densenet201_best.pth` |
| EffNet-B0 | 224 | 12 | Adam |  |  |  |  |  |  | `checkpoints/efficientnet_b0_best.pth` |
| EffNet-B1 | 240 | 12 | Adam |  |  |  |  |  |  | `checkpoints/efficientnet_b1_best.pth` |
| InceptionV3 | 299 | 12 | Adam |  |  |  |  |  | aux=0.4 | `checkpoints/inceptionv3_best.pth` |
| Inc-ResNet-v2 | 299 | 12 | Adam |  |  |  |  |  | timm | `checkpoints/inception_resnet_v2_best.pth` |
| ConvNeXt-T/S/B | 384 | 12 | AdamW |  |  |  |  |  | GA? | `checkpoints/convnext_384best.pth` |


---

## 7) GUI

- Load any `_best.pth`; reads `classes` from checkpoint.
- Single image or folder **batch** inference.
- Shows **top-k** probs and predicted class.
- Ensure **transforms match** the trained model’s input size (224/240/299/384).

Run:
```bash
python gui/app.py --checkpoint checkpoints/resnet50_best.pth --device cuda
