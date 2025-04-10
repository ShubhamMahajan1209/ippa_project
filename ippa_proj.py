import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Image Processing Functions
# def apply_noise_reduction(image):
#     return cv2.medianBlur(image, 5)

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def convert_to_black_and_white(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_edge_detection(image):
    return cv2.Canny(image, 100, 200)

def invert_colors(image):
    return cv2.bitwise_not(image)

def apply_gaussian_blur(image, ksize=5):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_smoothing(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Streamlit UI
st.title("Image Enhancement & Processing")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Ensure the image is in BGR format for OpenCV if it's RGB
    if image.shape[-1] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

    option = st.selectbox("Choose an enhancement technique", [
        "Image Sharpening",
        "Color to Black & White",
        "Edge Detection",
        "Invert Colors",
        "Gaussian Blur",
        "Smoothing"
    ])

    if st.button("Apply"):
        if option == "Image Sharpening":
            processed_image = apply_sharpening(image)
        elif option == "Color to Black & White":
            processed_image = convert_to_black_and_white(image)
        elif option == "Edge Detection":
            gray = convert_to_black_and_white(image)
            processed_image = apply_edge_detection(gray)
        elif option == "Invert Colors":
            processed_image = invert_colors(image)
        elif option == "Gaussian Blur":
            processed_image = apply_gaussian_blur(image)
        elif option == "Smoothing":
            processed_image = apply_smoothing(image)

        # Display processed image
        if len(processed_image.shape) == 2:
            st.image(processed_image, caption="Processed Image", use_column_width=True, channels="GRAY")
        else:
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
