import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Load the YOLO model
st.title("5S Audit Portal for Shop Floor")
model = YOLO("yolov8n.pt")  # Use the YOLOv8 model for object detection

st.header("5S Shop Floor Audit")
st.write("""
- **Sort**: Identify unnecessary items.
- **Set in Order**: Ensure tools are in designated places.
- **Shine**: Check for cleanliness.
- **Standardize**: Match with reference layout.
- **Sustain**: Ensure sustained adherence to 5S.
""")

# Reference image for comparison
st.sidebar.header("Upload Reference Image")
reference_file = st.sidebar.file_uploader("Upload Reference Layout", type=["jpg", "jpeg", "png"])

# Shop floor image for audit
st.header("Upload Shop Floor Image")
uploaded_file = st.file_uploader("Upload the current shop floor image for audit", type=["jpg", "jpeg", "png"])

if uploaded_file and reference_file:
    # Display uploaded images
    shop_floor_image = Image.open(uploaded_file)
    reference_image = Image.open(reference_file)

    st.image(shop_floor_image, caption="Current Shop Floor", use_column_width=True)
    st.image(reference_image, caption="Reference Layout", use_column_width=True)

    # Run object detection on the shop floor image
    st.write("Running object detection on shop floor image...")
    results = model.predict(np.array(shop_floor_image))

    # Object detection results
    detected_objects = []
    st.subheader("Detected Objects and Compliance Check")
    for result in results[0].boxes:
        obj_class = model.names[int(result.cls)]
        conf = result.conf.item()
        detected_objects.append((obj_class, conf))
        st.write(f"Object: {obj_class}, Confidence: {conf:.2f}")

    # Compliance Scoring (Example Logic)
    misplaced_items = ["item1", "item2"]  # Replace with actual 5S logic
    total_items = len(detected_objects)
    misplaced_count = len([item for item in detected_objects if item[0] in misplaced_items])
    compliance_score = max(0, 100 - (misplaced_count / total_items * 100))

    st.subheader("5S Audit Score")
    st.metric("Compliance Score", f"{compliance_score:.2f}%", delta=f"{-misplaced_count} deviations")

    # Annotated image for detected objects
    st.subheader("Detected Objects on Shop Floor")
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Objects", use_column_width=True)

else:
    st.info("Please upload both the reference layout and the shop floor image to perform the audit.")

