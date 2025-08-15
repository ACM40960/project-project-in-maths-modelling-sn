# Retinal Fundus Classification — Model Zoo (224/240/299/384)

Comprehensive pipeline for fundus-image classification with a consistent **80/10/10 stratified split**, a robust **preprocessing** stack, mixed-precision **training**, and a **GUI** for inference.  
> **Note:** We’ve **removed Swin** from the lineup. Added **ResNet-50/101** notebooks.

---

## 1) Overview

- Robust fundus preprocessing (ROI crop → shade correction → color constancy → CLAHE (G & L) → adaptive gamma → unsharp → letterbox+resize).
- Model zoo across **224/240/299/384** input sizes.
- **AMP** everywhere, checkpoint on **best val accuracy**, reproducible splits (`SEED=42`).
- Reports: **classification report**, **confusion matrix**, **ROC curves**, **AUC (macro/weighted/micro)**, **train/val curves**.

---

## 2) Data & Preprocessing

**Raw layout**
