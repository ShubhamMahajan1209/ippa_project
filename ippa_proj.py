import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Image Processing Functions

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

def apply_median_filter(image, ksize=3):
    return cv2.medianBlur(image, ksize)

def apply_mean_filter(image, ksize=3):
    return cv2.blur(image, (ksize, ksize))

def apply_min_filter(image, ksize=3):
    return cv2.erode(image, np.ones((ksize, ksize), np.uint8))

def apply_max_filter(image, ksize=3):
    return cv2.dilate(image, np.ones((ksize, ksize), np.uint8))

# Streamlit UI
st.title("Image Enhancement & Processing")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Convert to BGR if needed
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
        "Smoothing",
        "Median Filter",
        "Mean Filter",
        "Min Filter",
        "Max Filter"
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
            #kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
            processed_image = apply_gaussian_blur(image, ksize=7)
        elif option == "Smoothing":
            processed_image = apply_smoothing(image)
        elif option == "Median Filter":
           # kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
            processed_image = apply_median_filter(image, ksize=7)
        elif option == "Mean Filter":
           # kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
            processed_image = apply_mean_filter(image, ksize=7)
        elif option == "Min Filter":
            #kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
            processed_image = apply_min_filter(image, ksize=7)
        elif option == "Max Filter":
            #kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
            processed_image = apply_max_filter(image, ksize=7)

        # Show result
        if len(processed_image.shape) == 2:
            st.image(processed_image, caption="Processed Image", use_column_width=True, channels="GRAY")
        else:
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
