import os
import numpy as np
import pandas as pd
import cv2
import base64


def xray_image(base64_string):
    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_string)
    
    # Convert bytes to numpy array
    image_array = np.frombuffer(image_bytes, np.uint8)

    # Decode image
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image)

    # Step 5: Normalize final output for model input
    final = clahe_image / 255.0
    final.reshape(224, 224, 1)

    if final is None:
        raise ValueError("Failed to decode the image. Check the input format.")

    #cv2.imshow('web',image)
    #cv2.waitKey(0)

    return final

# Read Predicated labels
def Load_Predicated_ROI_Labels(txt_file_path):
    pred_bboxes = []

    # Read the TXT file line by line
    with open(txt_file_path, "r") as file:
        for line in file:
            # YOLO format: <class_id> <x_center> <y_center> <width> <height> <confidence>
            parts = line.strip().split()
            if len(parts) != 6:
                continue  # Skip invalid lines

            # Extract values from YOLO format
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            confidence = float(parts[5])

            # Convert YOLO format to (x_min, y_min, x_max, y_max)
            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            x_max = x_center + (width / 2)
            y_max = y_center + (height / 2)

            # Append the bounding box to the list
            pred_bboxes.append([x_min, y_min, x_max, y_max])

    # Convert the list to a NumPy array
    return np.array(pred_bboxes)

def create_cropped_image_test(image, bbox):
    """
    Crops ROI from a 224x224 CLAHE grayscale image using a YOLO-format bounding box.
    Returns a 4D tensor of shape (1, 224, 224, 1) suitable for model input.
    """
    # Validate input
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image input")

    if bbox is None or len(bbox) != 1 or len(bbox[0]) != 4:
        print("⚠️ Invalid bbox, returning full image resized")
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=-1)  # (224, 224, 1)
        return np.expand_dims(image, axis=0)    # (1, 224, 224, 1)

    # Image is 224x224, bbox in normalized format
    h, w = image.shape[:2]
    x_min = int(bbox[0][0] * w)
    y_min = int(bbox[0][1] * h)
    x_max = int(bbox[0][2] * w)
    y_max = int(bbox[0][3] * h)

    # Crop and resize
    cropped = image[y_min:y_max, x_min:x_max]
    cropped = cv2.resize(cropped, (224, 224))

    # Ensure shape (1, 224, 224, 1)
    if cropped.ndim == 2:
        cropped = np.expand_dims(cropped, axis=-1)
    roi_image = np.expand_dims(cropped, axis=0)

    return roi_image