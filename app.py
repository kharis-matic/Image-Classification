import streamlit as st
import cv2
import numpy as np
import joblib
import os
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import label, regionprops
from scipy.stats import skew, kurtosis

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mountain vs Glacier Classifier",
    page_icon="🏔️",
    layout="centered"
)

# ── Constants ──────────────────────────────────────────────────────────────────
FIXED_SIZE = (150, 150)
CLASS_NAMES = ['mountain', 'glacier']

# ── Helper Functions (identical to notebook) ────────────────────────────────────

def safe_stat(func, arr, default=0.0):
    arr = np.asarray(arr).astype(np.float32).ravel()
    if arr.size == 0:
        return float(default)
    try:
        val = func(arr)
        if np.isnan(val) or np.isinf(val):
            return float(default)
        return float(val)
    except:
        return float(default)


def read_image_from_array(img_rgb, size=FIXED_SIZE):
    img_rgb  = cv2.resize(img_rgb, size)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return img_rgb, img_gray, img_hsv, img_lab


def extract_color_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}
    rgb_names = ['r', 'g', 'b']
    for i, ch_name in enumerate(rgb_names):
        ch = img_rgb[:, :, i]
        feats[f'rgb_mean_{ch_name}'] = float(np.mean(ch))
        feats[f'rgb_std_{ch_name}']  = float(np.std(ch))
        feats[f'rgb_skew_{ch_name}'] = safe_stat(skew, ch)
    hsv_names = ['h', 's', 'v']
    for i, ch_name in enumerate(hsv_names):
        ch = img_hsv[:, :, i]
        feats[f'hsv_mean_{ch_name}'] = float(np.mean(ch))
        feats[f'hsv_std_{ch_name}']  = float(np.std(ch))
    lab_names = ['l', 'a', 'b']
    for i, ch_name in enumerate(lab_names):
        ch = img_lab[:, :, i]
        feats[f'lab_mean_{ch_name}'] = float(np.mean(ch))
        feats[f'lab_std_{ch_name}']  = float(np.std(ch))
    feats['gray_mean']     = float(np.mean(img_gray))
    feats['gray_std']      = float(np.std(img_gray))
    feats['gray_skew']     = safe_stat(skew,     img_gray)
    feats['gray_kurtosis'] = safe_stat(kurtosis, img_gray)
    hist_density, _ = np.histogram(img_gray, bins=256, range=(0,256), density=True)
    feats['gray_entropy'] = float(-np.sum(hist_density * np.log2(hist_density + 1e-12)))
    bins = 8
    for i, ch_name in enumerate(rgb_names):
        hist = cv2.calcHist([img_rgb], [i], None, [bins], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-12)
        for j, val in enumerate(hist):
            feats[f'rgb_hist_{ch_name}_{j}'] = float(val)
    blue_ch = img_rgb[:, :, 2].astype(float)
    red_ch  = img_rgb[:, :, 0].astype(float)
    feats['blue_red_ratio'] = float(np.mean(blue_ch) / (np.mean(red_ch) + 1e-12))
    feats['cool_pixel_fraction'] = float(np.mean(blue_ch > red_ch))
    feats['brightness_uniformity'] = float(1.0 / (np.std(img_gray) + 1e-12))
    return feats


def extract_glcm_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}
    glcm = graycomatrix(
        img_gray,
        distances=[1, 2],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )
    props = ['contrast', 'correlation', 'energy', 'homogeneity']
    for prop in props:
        values = graycoprops(glcm, prop).flatten()
        feats[f'glcm_{prop}_mean'] = float(np.mean(values))
        feats[f'glcm_{prop}_std']  = float(np.std(values))
    return feats


def extract_lbp_features(img_rgb, img_gray, img_hsv, img_lab, radius=1, n_points=8):
    feats = {}
    lbp    = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    n_bins = int(lbp.max()) + 1
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    for i, val in enumerate(hist):
        feats[f'lbp_hist_{i}'] = float(val)
    feats['lbp_mean'] = float(np.mean(lbp))
    feats['lbp_std']  = float(np.std(lbp))
    return feats


def create_dominant_region_mask(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest_idx, 255, 0).astype(np.uint8)
    return mask


def extract_shape_features(img_rgb):
    feats = {}
    mask  = create_dominant_region_mask(img_rgb)
    labeled = label(mask > 0)
    props   = regionprops(labeled)
    shape_keys = ['leaf_area','leaf_perimeter','leaf_bbox_w','leaf_bbox_h',
                  'leaf_aspect_ratio','leaf_extent','leaf_solidity',
                  'leaf_equiv_diameter','leaf_eccentricity']
    if len(props) == 0:
        for k in shape_keys:
            feats[k] = 0.0
        return feats
    region = max(props, key=lambda x: x.area)
    minr, minc, maxr, maxc = region.bbox
    bbox_h = maxr - minr
    bbox_w = maxc - minc
    feats['leaf_area']           = float(region.area)
    feats['leaf_perimeter']      = float(region.perimeter)
    feats['leaf_bbox_w']         = float(bbox_w)
    feats['leaf_bbox_h']         = float(bbox_h)
    feats['leaf_aspect_ratio']   = float(bbox_w / (bbox_h + 1e-12))
    feats['leaf_extent']         = float(region.extent)
    feats['leaf_solidity']       = float(region.solidity)
    feats['leaf_equiv_diameter'] = float(region.equivalent_diameter_area)
    feats['leaf_eccentricity']   = float(region.eccentricity)
    return feats


def extract_hog_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}
    hog_vec = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(32, 32),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    for i, val in enumerate(hog_vec):
        feats[f'hog_{i}'] = float(val)
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    feats['grad_mean'] = float(np.mean(grad_mag))
    feats['grad_std']  = float(np.std(grad_mag))
    return feats


def extract_all_features(img_rgb):
    img_rgb, img_gray, img_hsv, img_lab = read_image_from_array(img_rgb)
    feats = {}
    feats.update(extract_color_features(img_rgb, img_gray, img_hsv, img_lab))
    feats.update(extract_glcm_features(img_rgb, img_gray, img_hsv, img_lab))
    feats.update(extract_lbp_features(img_rgb, img_gray, img_hsv, img_lab))
    feats.update(extract_shape_features(img_rgb))
    feats.update(extract_hog_features(img_rgb, img_gray, img_hsv, img_lab))
    return feats


# ── Load Saved Model Artifacts ─────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load the saved model, imputer, scaler, and feature selectors."""
    required_files = [
        'best_model.pkl',
        'imputer.pkl',
        'scaler.pkl',
        'selector_stat.pkl',
        'top_idx.npy',
        'feature_names.pkl',
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        return None, missing

    model         = joblib.load('best_model.pkl')
    imputer       = joblib.load('imputer.pkl')
    scaler        = joblib.load('scaler.pkl')
    selector_stat = joblib.load('selector_stat.pkl')
    top_idx       = np.load('top_idx.npy')
    feature_names = joblib.load('feature_names.pkl')
    return {
        'model': model,
        'imputer': imputer,
        'scaler': scaler,
        'selector_stat': selector_stat,
        'top_idx': top_idx,
        'feature_names': feature_names,
    }, []


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏔️ Mountain vs Glacier Classifier")
st.markdown(
    "Upload an image and the model will predict whether it shows a **mountain** or a **glacier** "
    "using handcrafted image features (Color, GLCM, LBP, Shape, HOG)."
)
st.divider()

# Load artifacts
artifacts, missing_files = load_artifacts()

if missing_files:
    st.error("⚠️ The following saved model files were not found in the working directory:")
    for f in missing_files:
        st.code(f)
    st.info(
        "**How to generate them:** Add the following cell at the end of your Colab notebook "
        "and run it once after training, then download the files and place them alongside `app.py`."
    )
    st.code(
        """import joblib
import numpy as np

# Save after training — add this cell to the end of your notebook
joblib.dump(best_models[best_name],  'best_model.pkl')
joblib.dump(imputer,                 'imputer.pkl')
joblib.dump(scaler,                  'scaler.pkl')
joblib.dump(selector_stat,           'selector_stat.pkl')
np.save('top_idx.npy', top_idx)
joblib.dump(feature_names,           'feature_names.pkl')

print("All artifacts saved. Download them from the Colab file browser.")
""",
        language="python"
    )
    st.stop()

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Upload a landscape photo. The model will classify it as glacier or mountain."
)

if uploaded_file is not None:
    # Display uploaded image
    pil_image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(pil_image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Extracting features and predicting..."):
            # Convert to numpy array (RGB)
            img_rgb = np.array(pil_image)

            # Extract features
            feats_dict = extract_all_features(img_rgb)
            all_feature_names = artifacts['feature_names']

            # Build feature vector (in the same column order as training)
            feat_vector = np.array(
                [feats_dict.get(name, 0.0) for name in all_feature_names],
                dtype=np.float32
            ).reshape(1, -1)

            # Preprocess
            feat_imp  = artifacts['imputer'].transform(feat_vector)
            feat_sc   = artifacts['scaler'].transform(feat_imp)      # 408 features
            feat_sel  = feat_sc[:, artifacts['top_idx']]     

            # Predict
            model      = artifacts['model']
            prediction = model.predict(feat_sel)[0]
            label_name = CLASS_NAMES[prediction]

            # Confidence (if model supports predict_proba)
            confidence = None
            confidence_label = None

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feat_sel)[0]
                confidence = proba[prediction]
                confidence_label = "Probability"
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function(feat_sel)[0]
                # Convert decision score to 0-1 range using sigmoid
                confidence = float(1 / (1 + np.exp(-abs(decision))))
                confidence_label = "Confidence Score (sigmoid)"

        # Results
        emoji = "🧊" if label_name == "glacier" else "⛰️"
        st.markdown(f"### Prediction: {emoji} `{label_name.upper()}`")

        if confidence is not None:
            st.metric(confidence_label, f"{confidence * 100:.1f}%")
            
            # Color-coded progress bar via markdown
            bar_color = "#4FC3F7" if label_name == "glacier" else "#8D6E63"
            st.progress(float(confidence))
            
            # Show both class probabilities if available
            if hasattr(model, 'predict_proba'):
                st.markdown("**Class Probabilities:**")
                for i, cls in enumerate(CLASS_NAMES):
                    emoji_cls = "🧊" if cls == "glacier" else "⛰️"
                    st.write(f"{emoji_cls} {cls.capitalize()}: `{proba[i]*100:.1f}%`")
                    st.progress(float(proba[i]))
            else:
                st.markdown("**Decision confidence:**")
                st.write(f"{emoji} {label_name.capitalize()}: `{confidence*100:.1f}%`")
                st.progress(float(confidence))
        else:
            st.info("This model does not support confidence estimation.")

        st.divider()

        # Feature highlights
        st.markdown("**Key features detected:**")
        hsv_h   = feats_dict.get('hsv_mean_h', 0)
        blue_r  = feats_dict.get('blue_red_ratio', 0)
        solidity = feats_dict.get('leaf_solidity', 0)
        gray_std = feats_dict.get('gray_std', 0)

        st.write(f"- HSV Mean Hue: `{hsv_h:.2f}`")
        st.write(f"- Blue/Red Ratio: `{blue_r:.2f}`")
        st.write(f"- Region Solidity: `{solidity:.2f}`")
        st.write(f"- Grayscale Std Dev: `{gray_std:.2f}`")

st.divider()
st.caption(
    "Model: handcrafted features (Color · GLCM · LBP · Shape · HOG) + classical ML classifier · "
    "Intel Image Classification dataset (glacier / mountain)"
)
