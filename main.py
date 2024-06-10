import streamlit as st
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image

pytesseract.pytesseract.tesseract_cmd ='tesseract'
tessdata_dir_config = '--tessdata-dir "."

def preprocess_image(image_cv):
    # Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    return sharpened

def detect_and_highlight_barcode(image):
    image_cv = np.array(image)
    barcode_data_list = []

    preprocessed_image = preprocess_image(image_cv)
    
    barcodes = decode(preprocessed_image)
    
    for barcode in barcodes:
        # Extract the bounding box coordinates
        x, y, w, h = barcode.rect
        # Crop the barcode from the image
        barcode_crop = image_cv[y:y + h, x:x + w]
        
        # Convert to PIL format for any additional preprocessing if needed
        pil_image = Image.fromarray(barcode_crop)
        
        # Convert back to OpenCV format
        barcode_crop = np.array(pil_image)
        
        # Highlight the barcode with a thick rectangle
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 10)  # Green rectangle with thickness 10
        
        # Extract and store the barcode data
        barcode_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type
        
        barcode_data_list.append((barcode_data, barcode_type))
    
    return image_cv, barcode_data_list

st.title("Barcode Scanner App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    
    # Detect and highlight the barcode
    result_image, barcode_data_list = detect_and_highlight_barcode(image)
    
    # Convert the result image to PIL format for display
    result_image_pil = Image.fromarray(result_image)
    
    # Display the original and result images
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(result_image_pil, caption="Processed Image with Barcode Highlighted", use_column_width=True)
    
    # Display barcode data
    st.subheader("Detected Barcodes")
    for barcode_data, barcode_type in barcode_data_list:
        st.write(f"Type: {barcode_type}, Data: {barcode_data}")

    # Convert the PIL image to bytes for downloading
    import io
    buf = io.BytesIO()
    result_image_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    # Provide download button for the processed image
    st.download_button(
        label="Download Processed Image",
        data=byte_im,
        file_name="processed_image.png",
        mime="image/png"
    )
