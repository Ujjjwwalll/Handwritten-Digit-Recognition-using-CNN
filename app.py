
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from io import BytesIO

# -----------------------------
# 1. Page Configuration
# -----------------------------

st.set_page_config(
    page_title="CNN Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognition")
st.write("Upload a handwritten digit image and let our CNN predict it.")

# -----------------------------
# 2. Load Trained CNN Model
# -----------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.keras")

model = load_model()

# -----------------------------
# 3. Image Preprocessing
# -----------------------------


def preprocess_image(image):

    # Convert uploaded image to RGB
    img = np.array(image.convert("RGB"))

    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Extract saturation channel (useful for blue ink)
    saturation = hsv[:, :, 1]

    # Detect colored ink
    _, color_mask = cv2.threshold(
        saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Grayscale fallback (useful for black ink)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, dark_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Select the more suitable mask
    pixel_ratio = np.count_nonzero(color_mask) / color_mask.size

    if 0.0005 < pixel_ratio < 0.30:
        thresh = color_mask
    else:
        thresh = dark_mask

    # Connect small gaps in strokes
    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Make thin strokes slightly thicker
    thresh = cv2.dilate(
        thresh,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No digit detected.")

    # Find largest contour
    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < 10:
        raise ValueError("No clear digit detected.")

    # Crop digit
    x, y, w, h = cv2.boundingRect(contour)

    digit = thresh[y:y+h, x:x+w]

    # Resize while maintaining aspect ratio
    scale = 20 / max(w, h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Create 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # Center digit
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = resized

    # Increase stroke thickness
    kernel = np.ones((3, 3), np.uint8)

    canvas = cv2.dilate(
        canvas,
        kernel,
        iterations=1
    )

# Normalize
    processed = canvas.astype("float32") / 255.0
    

    # Reshape for CNN
    processed = processed.reshape(1, 28, 28, 1)

    return processed, canvas



# -----------------------------
# 4. Upload Image
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload your handwritten digit",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# 5. Make Prediction
# -----------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, width=250)

    if st.button("Predict Digit"):

        try:
            processed_img, display_img = preprocess_image(image)

            prediction = model.predict(
                processed_img,
                verbose=0
            )

            predicted_digit = int(np.argmax(prediction[0]))

            confidence = float(np.max(prediction[0])) * 100

            st.subheader("Prediction Result")

            st.success(f"Predicted Digit: {predicted_digit}")

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.subheader("Processed Image (28 × 28)")
            st.image(display_img, width=150)

            st.subheader("Class Probabilities")

            st.bar_chart({
                str(i): float(prediction[0][i])
                for i in range(10)
            })

        except ValueError as e:
            st.error(str(e))

        except Exception as e:
            st.error(f"Prediction failed: {e}")
