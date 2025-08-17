# Streamlit GUI for fundus image classification using the best model: Inception-ResNet-v2
import streamlit as st
import torch
import timm
import numpy as np
import cv2
from PIL import Image
import os
import gdown

# ===================== CONFIGURATION =====================
# Path to the trained model checkpoint
FILE_ID = "1_ZWh1IkXRnenXtrqoRr73QYrv-bjNkWL"
MODEL_PATH = "inception_resnet_v2_best.pth"

# Class names for fundus classification
CLASS_NAMES = ['ARMD', 'cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Use GPU if available, else fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== DOWNLOAD MODEL FROM DRIVE =====================
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    st.write("Downloading model checkpoint from Google Drive...")
    gdown.download(url=url, output=MODEL_PATH, fuzzy=True)
    
# ===================== PREPROCESS FUNCTIONS =====================
def imread_any_bytes(file_bytes):
    """Read image bytes (uploaded file) and convert to OpenCV BGR image."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def robust_bbox_from_mask(mask, margin_ratio=0.02):
    """Calculate a bounding box around non-zero pixels of a mask, with margin."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    x, y, w, h = cv2.boundingRect(coords)
    m = int(max(h, w) * margin_ratio)
    x = max(0, x - m)
    y = max(0, y - m)
    return x, y, min(mask.shape[1] - x, w + 2*m), min(mask.shape[0] - y, h + 2*m)


def fundus_roi_crop(bgr):
    """Crop the fundus region from the image using a circular mask."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    x, y, w, h = robust_bbox_from_mask(mask)
    return bgr[y:y+h, x:x+w]


def shade_correction(bgr, sigma=40):
    """Correct illumination/shading by dividing by Gaussian-blurred background."""
    I = bgr.astype(np.float32) + 1.0
    bg = cv2.GaussianBlur(I, (0,0), sigmaX=sigma, sigmaY=sigma)
    corrected = I / (bg + 1e-6) * 128.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def shades_of_gray_cc(bgr, p=6, eps=1e-6):
    """Apply Shades-of-Gray color constancy."""
    I = bgr.astype(np.float32)
    Rp = np.power(I[:,:,2], p).mean() ** (1.0/p)
    Gp = np.power(I[:,:,1], p).mean() ** (1.0/p)
    Bp = np.power(I[:,:,0], p).mean() ** (1.0/p)
    scale = (Rp + Gp + Bp) / 3.0
    R = I[:,:,2] * (scale / (Rp + eps))
    G = I[:,:,1] * (scale / (Gp + eps))
    B = I[:,:,0] * (scale / (Bp + eps))
    out = np.stack([B, G, R], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_on_green(bgr, clip=2.0, tile=(8,8)):
    """Apply CLAHE on the green channel."""
    b, g, r = cv2.split(bgr)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    g2 = clahe.apply(g)
    return cv2.merge([b, g2, r])


def optional_l_channel_clahe(bgr, clip=1.5, tile=(8,8)):
    """Apply CLAHE on L-channel of LAB color space."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def adaptive_gamma(bgr, target=0.42):
    """Adjust gamma based on image median."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    med = np.median(gray) / 255.0
    med = np.clip(med, 1e-4, 0.999)
    gamma = np.log(target) / np.log(med)
    inv = 1.0 / np.clip(gamma, 0.2, 5.0)
    lut = np.arange(256, dtype=np.float32) / 255.0
    lut = np.power(lut, inv)
    lut = np.clip(lut * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(bgr, lut)


def unsharp(bgr, sigma=1.0, amount=0.5):
    """Apply unsharp masking to enhance details."""
    blurred = cv2.GaussianBlur(bgr, (0,0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(bgr, 1 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def letterbox_square(bgr, size=299, color=(0,0,0)):
    """Pad to square and resize to target size."""
    h, w = bgr.shape[:2]
    dim = max(h, w)
    top = (dim - h) // 2
    bottom = dim - h - top
    left = (dim - w) // 2
    right = dim - w - left
    padded = cv2.copyMakeBorder(bgr, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return cv2.resize(padded, (size, size), interpolation=cv2.INTER_CUBIC)


def preprocess_fundus(bgr, size=299):
    """Full preprocessing pipeline for fundus images."""
    img = fundus_roi_crop(bgr)
    img = shade_correction(img)
    img = shades_of_gray_cc(img)
    img = clahe_on_green(img)
    img = optional_l_channel_clahe(img)
    img = adaptive_gamma(img)
    img = unsharp(img)
    img = letterbox_square(img, size=size)
    return img


# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    """Load the Inception-ResNet-v2 model from checkpoint."""
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = checkpoint["classes"]

    model = timm.create_model("inception_resnet_v2",
                              pretrained=False,
                              num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()
    return model, classes


# Load model
model, CLASS_NAMES = load_model()


# ===================== PREDICTION FUNCTION =====================
def predict_fundus(image_bgr):
    """Preprocess image, run inference, return predicted class and confidence."""
    img = preprocess_fundus(image_bgr, size=299)
    img = img[:, :, ::-1]  # BGR → RGB
    tensor = torch.tensor(img / 255.0, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        if isinstance(outputs, tuple):  # handle aux logits
            outputs = outputs[0]
        probs = torch.softmax(outputs, dim=1)[0]
        top_idx = torch.argmax(probs).item()

    return CLASS_NAMES[top_idx], probs[top_idx].item()


# ===================== STREAMLIT GUI =====================
st.title("🩺 OcuScan - Fundus Image Classifier (Inception-ResNet v2)")
st.write("Upload a fundus image for diagnosis.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    bgr_image = imread_any_bytes(uploaded_file.read())
    st.image(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB),
             caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        label, confidence = predict_fundus(bgr_image)
        st.success(f"Prediction: **{label}** (Confidence: {confidence:.2%})")
