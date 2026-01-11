import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import math

st.set_page_config(layout="wide")
st.title("Advanced Recursive Shape Analyzer")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

MAX_DEPTH = 1  # prevents infinite recursion

def classify_shape(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if area < 200 or peri == 0:
        return None

    epsilon = 0.03 * peri  # IMPORTANT: larger epsilon
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    v = len(approx)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else 0

    circularity = 4 * math.pi * area / (peri * peri + 1e-6)

    # -------- TRIANGLE (robust) --------
    if 3 <= v <= 5 and solidity > 0.8:
        return "Triangle"

    # -------- QUADRILATERALS --------
    if v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        rect = cv2.minAreaRect(cnt)
        angle = abs(rect[2])

        if solidity > 0.95 and 0.9 < ar < 1.1:
            return "Square"
        if solidity > 0.9:
            return "Rectangle"

    # -------- CIRCLE --------
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    circle_area = math.pi * radius * radius
    if radius > 5 and abs(circle_area - area) / circle_area < 0.2 and circularity > 0.85:
        return "Circle"

    # -------- OVAL --------
    if len(cnt) >= 8 and circularity < 0.85:
        ellipse = cv2.fitEllipse(cnt)
        (_, _), (MA, ma), _ = ellipse
        ratio = min(MA, ma) / max(MA, ma)
        if 0.5 < ratio < 0.85:
            return "Oval"

    # -------- POLYGON --------
    if v >= 5:
        return "Polygon"

    return None


def recursive_detect(img, offset_x=0, offset_y=0, depth=0, results=None, canvas=None):
    if depth > MAX_DEPTH:
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blur, 60, 160)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 150:
            continue

        shape = classify_shape(cnt)
        if shape is None:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        results.append({
            "Shape": shape,
            "Area": int(area),
            "Perimeter": int(cv2.arcLength(cnt, True))
        })

        cv2.rectangle(
            canvas,
            (offset_x + x, offset_y + y),
            (offset_x + x + w, offset_y + y + h),
            (0, 255, 0),
            1
        )
        cv2.putText(
            canvas,
            shape,
            (offset_x + x, offset_y + y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1
        )

        # --- Recursive inspection inside contour ---
        roi = img[y:y+h, x:x+w]
        if roi.size > 0:
            recursive_detect(
                roi,
                offset_x + x,
                offset_y + y,
                depth + 1,
                results,
                canvas
            )

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    canvas = img.copy()
    results = []

    recursive_detect(img, results=results, canvas=canvas)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detected Shapes (Recursive)")
        st.image(canvas, use_container_width=True)

    df = pd.DataFrame(results)
    st.subheader("Detected Shape Summary")
    st.dataframe(df)

    st.success(f"Total Shapes Detected: {len(df)}")
