import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyzbar.pyzbar import decode
import pytesseract
import re
import streamlit as st

def process_image(image_path):
    img = Image.open(image_path)

    if img.height > img.width:
        img = img.rotate(90, expand=True)

    draw = ImageDraw.Draw(img)
    font_path = 'simfang.ttf'
    font = ImageFont.truetype(font_path, size=20)

    detected_texts = []

    for d in decode(img):
        if d.type != 'QRCODE':
            barcode_data = d.data.decode()
            draw.rectangle(((d.rect.left, d.rect.top), (d.rect.left + d.rect.width, d.rect.top + d.rect.height)),
                           outline=(0, 0, 255), width=3)
            buffer = 10
            text_region = (d.rect.left, d.rect.top + d.rect.height + buffer,
                           d.rect.left + d.rect.width, d.rect.top + d.rect.height + buffer + 50)
            text_image = img.crop(text_region)
            text_image_cv = np.array(text_image)
            text_image_cv = cv2.cvtColor(text_image_cv, cv2.COLOR_RGB2GRAY)
            custom_config = r'--oem 3 --psm 6'
            detected_text = pytesseract.image_to_string(text_image_cv, config=custom_config).strip()
            filtered_text = re.sub('[^A-Z0-9]', '', detected_text)
            draw.text((d.rect.left, d.rect.top + d.rect.height + buffer), detected_text, (255, 0, 0), font=font)
            detected_texts.append((detected_text, filtered_text))

    open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img, detected_texts, open_cv_image

st.title("Barcode and Text Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    with st.spinner('Processing...'):
        img, detected_texts, open_cv_image = process_image(uploaded_file)

        st.image(img, caption='Processed Image', use_column_width=True)

        st.subheader("Detected Texts")
        for i, (detected_text, filtered_text) in enumerate(detected_texts):
            st.write(f"Barcode Text is: {filtered_text}")

        st.subheader("OpenCV Image")
        st.image(open_cv_image, channels="BGR", use_column_width=True)
