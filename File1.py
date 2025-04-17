import os
import numpy as np
import json
import cv2
from pathlib import Path
import tempfile
import torch
import tensorflow as tf
from yolov5 import detect
from tensorflow.keras.models import load_model
from supporting_functions import create_cropped_image_test,Load_Predicated_ROI_Labels
import glob
from supporting_functions import xray_image
from utils.blob_model_loader import download_model_if_needed

def test_yolo(image_input, model_path, output_dir):
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    temp_path = None

    # ✅ Step 1: Process grayscale normalized image (shape = 224x224, float32)
    if isinstance(image_input, np.ndarray):
        # Convert from [0.0–1.0] float to [0–255] uint8
        if image_input.max() <= 1.0:
            image_input = (image_input * 255).astype(np.uint8)

        # Ensure 3-channel for YOLO
        if image_input.ndim == 2:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
        cv2.imwrite(temp_path, image_input)
        image_path = temp_path

    elif isinstance(image_input, (str, Path)):
        image_path = Path(image_input)

    else:
        raise ValueError("Unsupported image input format")

    print(f"📸 Running detection on: {image_path}")

    # ✅ Step 2: Run YOLOv5 Detection
    detect.run(
        weights=str(model_path),
        source=str(image_path),
        conf_thres=0.5,
        iou_thres=0.6,
        imgsz=(224, 224),
        save_txt=True,
        save_conf=True,
        project=str(output_dir),
        exist_ok=True
    )

    print(f"✅ Detection complete. Results saved to: {output_dir/'exp'}")

    # ✅ Step 3: Clean up temp image
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    # ✅ Step 4: Read label file
    predictions_dir = output_dir / 'exp' / 'labels'
    print(f"🔍 Looking for labels in: {predictions_dir}")
    label_file_paths = glob.glob(str(predictions_dir / '*.txt'))

    if not label_file_paths:
        print("❌ No label files found.")
        return None

    print(f"✅ Found label file: {label_file_paths[0]}")
    bounding_boxes = Load_Predicated_ROI_Labels(label_file_paths[0])
    print("📦 Bounding box details:", bounding_boxes)

    return bounding_boxes



def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def bmd_predict(roi_image, features, bmd_model_path):
    # Load the model with the custom object
    bmd_model = load_model(bmd_model_path, custom_objects={'mse': mse})
    bmd_value = bmd_model.predict([roi_image, features]).flatten()

    return bmd_value

def T_score_predict(roi_image, features,T_score_model_path, json_path):
    # Load the trained T-Score model
    # Load the model with the custom object
    T_Score_model = load_model(T_score_model_path, custom_objects={'mse': mse})

    # Load normalization parameters from the JSON file
    with open(json_path, "r") as f:
        normalization_params = json.load(f)

    T_Score_min = normalization_params["T_Score_min"]
    T_Score_max = normalization_params["T_Score_max"]

    # Predict normalized T-scores
    T_Score_normalized = T_Score_model.predict([roi_image, features]).flatten()

    print("T-score normalized:",T_Score_normalized)
    # Denormalize predicted T-scores to the original range
    T_Score_value = (
        T_Score_normalized * (T_Score_max - T_Score_min)
    ) + T_Score_min

    return T_Score_value

def Z_score_predict(roi_image, features, Z_score_model_path, json_path):
    # Load the model with the custom object
    Z_score_model = load_model(Z_score_model_path, custom_objects={'mse': mse})

    # Load normalization parameters from the JSON file
    with open(json_path, "r") as f:
        normalization_params = json.load(f)

    Z_Score_min = normalization_params["Z_Score_min"]
    Z_Score_max = normalization_params["Z_Score_max"]

    # Predict normalized T-scores
    Z_Score_normalized = Z_score_model.predict([roi_image, features]).flatten()

    # Denormalize predicted T-scores to the original range
    Z_Score_value = (
        Z_Score_normalized * (Z_Score_max - Z_Score_min)
    ) + Z_Score_min

    return Z_Score_value


def WHO_predict(roi_image, features, WHO_model_path):
    # Load the model with the custom object
    WHO_model = load_model(WHO_model_path, custom_objects={'mse': mse})

    # Predict the probabilities (assuming it's a classification problem)
    WHO_probs = WHO_model.predict([roi_image, features])

    # If the model outputs probabilities, take the class with the highest probability
    WHO_value = np.argmax(WHO_probs)  # This will return the index of the highest probability

    return WHO_value


def testing_models(predicated_bbox_output_file,yolo_model_path,bmd_model_path,T_score_model_path,Z_score_model_path,WHO_model_path,T_score_json_path,Z_score_json_path,Test_xray_image,Test_Features):

    predicated_bboxes = test_yolo(Test_xray_image,yolo_model_path,predicated_bbox_output_file)
   
    cropped_predicated_image = create_cropped_image_test(Test_xray_image, predicated_bboxes)
   
    # Define column indices for each feature subset
    bmd_indices = [0, 2, 3, 4, 5]        # age, bmi, blood_group_encoded, potassium, alkaline_phosphatase
    z_score_indices = [0, 1, 3, 4, 5]    # age, weight, blood_group_encoded, potassium, alkaline_phosphatase
    t_who_indices = [0, 1, 4, 5]         # age, weight, potassium, alkaline_phosphatase

    # Extract subsets using advanced indexing
    bmd_features = Test_Features[:, bmd_indices]
    z_score_features = Test_Features[:, z_score_indices]
    t_who_features = Test_Features[:, t_who_indices]

    bmd = bmd_predict(cropped_predicated_image, bmd_features, bmd_model_path)
    T_score = T_score_predict(cropped_predicated_image, t_who_features, T_score_model_path, T_score_json_path)
    Z_score = Z_score_predict(cropped_predicated_image, z_score_features, Z_score_model_path, Z_score_json_path)
    WHO_classification = WHO_predict(cropped_predicated_image, t_who_features, WHO_model_path)
    Fracture_risk=''


    if (WHO_classification == 0):
        Fracture_risk='NOT INCREASED'
    elif (WHO_classification == 1):
        Fracture_risk='INCREASED'
    elif (WHO_classification == 2):
        Fracture_risk='HIGH'
        

    return bmd,T_score,Z_score,WHO_classification,Fracture_risk   

 
def integration(file_path, Features):
    model_dir = "models"
    config_dir = "config"
    os.makedirs(model_dir, exist_ok=True)

    # ✅ Step 1: Download models from Azure Blob if not already present
    download_model_if_needed("yolo.pt", f"{model_dir}/yolo.pt")
    download_model_if_needed("bmd_model.h5", f"{model_dir}/bmd_model.h5")
    download_model_if_needed("T_score_model.h5", f"{model_dir}/T_score_model.h5")
    download_model_if_needed("Z_score_model.h5", f"{model_dir}/Z_score_model.h5")
    download_model_if_needed("WHO_model.h5", f"{model_dir}/WHO_model.h5")

    # ✅ Step 2: Download JSON configs (optional but recommended)
    os.makedirs(config_dir, exist_ok=True)
    download_model_if_needed("T_Score_normalization_params.json", f"{config_dir}/T_Score_normalization_params.json")
    download_model_if_needed("Z_Score_normalization_params.json", f"{config_dir}/Z_Score_normalization_params.json")

    # ✅ Step 3: Set paths as usual
    predicated_bbox_output_file = "static"
    yolo_model_path = f"{model_dir}/yolo.pt"
    bmd_model_path = f"{model_dir}/bmd_model.h5"
    T_score_model_path = f"{model_dir}/T_score_model.h5"
    Z_score_model_path = f"{model_dir}/Z_score_model.h5"
    WHO_model_path = f"{model_dir}/WHO_model.h5"
    T_score_json_path = f"{config_dir}/T_Score_normalization_params.json"
    Z_score_json_path = f"{config_dir}/Z_Score_normalization_params.json"

    # ✅ Step 4: Continue as usual
    bmd,T_score,Z_score,WHO_classification,Fracture_risk = testing_models(
        predicated_bbox_output_file,
        yolo_model_path,
        bmd_model_path,
        T_score_model_path,
        Z_score_model_path,
        WHO_model_path,
        T_score_json_path,
        Z_score_json_path,
        file_path,
        Features
    )
    return bmd,T_score,Z_score,WHO_classification,Fracture_risk 




