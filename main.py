import streamlit as st
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image
import pytesseract

def preprocess_image(image_cv):
    # Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
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
        
        # Highlight the barcode with a thick rectangle
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 10)  # Green rectangle with thickness 10
        
        # Extract and store the barcode data
        barcode_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type
        
        barcode_data_list.append((barcode_data, barcode_type))
        
        # Define the region below the barcode to search for text
        padding = 10
        text_region_y_start = y + h + padding
        text_region_y_end = y + h + padding + 40

        # Ensure the region is within image boundaries
        if text_region_y_start < image_cv.shape[0] and text_region_y_end <= image_cv.shape[0] and x < image_cv.shape[1] and x + w <= image_cv.shape[1]:
            text_region = image_cv[text_region_y_start:text_region_y_end, x:x + w]
            
            # Debugging: Log the region coordinates
            print(f'Text region coordinates: x={x}, y_start={text_region_y_start}, y_end={text_region_y_end}, w={w}')
            
            # Convert the text region to grayscale
            text_region_gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
            
            # Use pytesseract to extract text from the region
            text_data = pytesseract.image_to_string(text_region_gray, config='--psm 7').strip()
            
            if text_data:
                barcode_data_list.append((text_data, "Text"))
                # Draw a rectangle around the detected text
                cv2.rectangle(image_cv, (x, text_region_y_start), (x + w, text_region_y_end), (255, 0, 0), 10)  # Blue rectangle
        else:
            # Debugging: Log the reason why the region was not processed
            print(f'Out of bounds: x={x}, y_start={text_region_y_start}, y_end={text_region_y_end}, w={w}, img_height={image_cv.shape[0]}, img_width={image_cv.shape[1]}')
    
    return image_cv, barcode_data_list

def rotate_image_if_vertical(image):
    if image.height > image.width:
        image = image.rotate(90, expand=True)
    return image

st.title("Barcode Scanner App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    
    # Rotate the image if it's vertical
    image = rotate_image_if_vertical(image)
    
    # Detect and highlight the barcode
    result_image, barcode_data_list = detect_and_highlight_barcode(image)
    
    # Convert the result image to PIL format for display
    result_image_pil = Image.fromarray(result_image)
    
    # Display the original and result images
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(result_image_pil, caption="Processed Image with Barcode Highlighted", use_column_width=True)
    
    # Display barcode data
    st.subheader("Detected Barcodes and Text")
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
