
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from scipy.ndimage import shift
from streamlit_drawable_canvas import st_canvas


# ----------------------------------
# 1. Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="MNIST Studio",
    page_icon="✍️",
    layout="wide"
)


# ----------------------------------
# 2. Load Model
# ----------------------------------

MODEL_PATH = Path(__file__).with_name("mnist_svm_model.pkl")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    if model.n_features_in_ != 784:
        raise ValueError("Model must have 784 features.")

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba().")

    return model


try:
    model = load_model()
except Exception as error:
    st.error(f"Model loading error: {error}")
    st.stop()


# ----------------------------------
# 3. Image Preprocessing
# ----------------------------------

def prepare_digit(gray_image):

    image = np.array(gray_image, dtype=np.float32)

    rows, cols = np.where(image > 30)

    if len(rows) == 0:
        return None

    # Crop digit
    image = image[
        rows.min():rows.max() + 1,
        cols.min():cols.max() + 1
    ]

    # Resize while maintaining aspect ratio
    height, width = image.shape
    scale = 20 / max(height, width)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    digit = Image.fromarray(image)

    resampling = getattr(Image, "Resampling", Image)

    digit = digit.resize(
        (new_width, new_height),
        resampling.LANCZOS
    )

    digit = np.array(digit, dtype=np.float32)

    # Place inside 28x28 image
    canvas = np.zeros((28, 28), dtype=np.float32)

    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = digit

    # Center the digit
    total = canvas.sum()

    if total > 0:

        rows, cols = np.indices(canvas.shape)

        center_y = (rows * canvas).sum() / total
        center_x = (cols * canvas).sum() / total

        shift_y = int(round(13.5 - center_y))
        shift_x = int(round(13.5 - center_x))

        canvas = shift(
            canvas,
            (shift_y, shift_x),
            order=1,
            mode="constant",
            cval=0
        )

    # Normalize and reshape
    canvas = np.clip(canvas, 0, 255) / 255.0

    return canvas.reshape(1, 784)


# ----------------------------------
# 4. Uploaded Image Processing
# ----------------------------------

def uploaded_image_to_gray(uploaded_file):

    image = Image.open(uploaded_file).convert("L")

    image_array = np.array(image)

    if image_array.mean() > 127:
        image_array = 255 - image_array

    return image_array


# ----------------------------------
# 5. Prediction Display
# ----------------------------------

def show_prediction(gray_image):

    if gray_image is None:
        st.info("Draw or upload a digit to see its prediction.")
        return

    model_input = prepare_digit(gray_image)

    if model_input is None:
        st.warning("No digit detected. Please try again.")
        return

    probabilities = model.predict_proba(model_input)[0]

    best_index = int(np.argmax(probabilities))

    predicted_digit = int(model.classes_[best_index])

    confidence = float(probabilities[best_index])

    st.subheader("Prediction Result")

    st.metric(
        "Predicted Digit",
        predicted_digit
    )

    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )

    st.divider()

    # Top 3 predictions
    st.subheader("Top 3 Predictions")

    top_indices = np.argsort(probabilities)[::-1][:3]

    for index in top_indices:

        digit = int(model.classes_[index])

        probability = float(probabilities[index])

        st.write(f"Digit {digit}: {probability * 100:.2f}%")

        st.progress(probability)

    st.divider()

    # All probabilities
    st.subheader("All Digit Probabilities")

    probability_df = pd.DataFrame({
        "Digit": [str(int(value)) for value in model.classes_],
        "Probability": probabilities
    })

    st.bar_chart(
        probability_df.set_index("Digit")
    )


# ----------------------------------
# 6. App Header
# ----------------------------------

st.title("✍️ MNIST Studio")

st.subheader("Handwritten Digit Recognition")

st.write(
    "Draw a digit from 0 to 9 or upload an image. "
    "Our trained SVM model will recognize your digit."
)

st.divider()


# ----------------------------------
# 7. Main Application Layout
# ----------------------------------

draw_tab, upload_tab = st.tabs([
    "✏️ Draw a Digit",
    "📁 Upload an Image"
])


# ----------------------------------
# 8. Drawing Tab
# ----------------------------------

with draw_tab:

    left_col, right_col = st.columns(
        [1, 1],
        gap="large"
    )

    with left_col:

        st.subheader("Draw Your Digit")

        st.write("Draw one digit using your mouse.")

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=15,
            stroke_color="#FFFFFF",
            background_color="#000000",
            drawing_mode="freedraw",
            height=360,
            width=360,
            display_toolbar=True,
            update_streamlit=True,
            key="digit_canvas"
        )

        gray_image = None

        if canvas_result.image_data is not None:

            canvas_image = Image.fromarray(
                canvas_result.image_data.astype(np.uint8)
            ).convert("L")

            gray_image = np.array(
                canvas_image,
                dtype=np.float32
            )

    with right_col:

        st.subheader("Recognition")

        show_prediction(gray_image)


# ----------------------------------
# 9. Upload Tab
# ----------------------------------

with upload_tab:

    left_col, right_col = st.columns(
        [1, 1],
        gap="large"
    )

    with left_col:

        st.subheader("Upload Your Image")

        uploaded_file = st.file_uploader(
            "Choose a handwritten digit image",
            type=["png", "jpg", "jpeg"]
        )

        gray_image = None

        if uploaded_file is not None:

            try:

                gray_image = uploaded_image_to_gray(
                    uploaded_file
                )

                st.image(
                    uploaded_file,
                    caption="Uploaded Image",
                    width=250
                )

            except Exception as error:

                st.error(f"Image error: {error}")

    with right_col:

        st.subheader("Recognition")

        show_prediction(gray_image)


# ----------------------------------
# 10. Footer
# ----------------------------------

st.divider()

footer_col1, footer_col2 = st.columns(2)

with footer_col1:

    st.caption("MNIST Handwritten Digit Recognition")

    st.caption("Machine Learning | Multi-class Classification | SVM")

with footer_col2:

    st.caption("Developed by Erum Shabir")

    st.caption("AI & Data Science Portfolio Project")
