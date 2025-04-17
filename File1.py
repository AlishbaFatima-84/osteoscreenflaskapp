import os
import numpy as np
import json
import cv2
from pathlib import Path
import tempfile
import glob
from supporting_functions import create_cropped_image_test, Load_Predicated_ROI_Labels
from utils.blob_model_loader import download_model_if_needed


def test_yolo(image_input, model_path, output_dir):
    from yolov5 import detect  # ✅ Lazy load

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    temp_path = None

    if isinstance(image_input, np.ndarray):
        if image_input.max() <= 1.0:
            image_input = (image_input * 255).astype(np.uint8)

        if image_input.ndim == 2:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
        cv2.imwrite(temp_path, image_input)
        image_path = temp_path

    elif isinstance(image_input, (str, Path)):
        image_path = Path(image_input)

    else:
        raise ValueError("Unsupported image input format")

    print(f"📸 Running detection on: {image_path}")

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

    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    predictions_dir = output_dir / 'exp' / 'labels'
    label_file_paths = glob.glob(str(predictions_dir / '*.txt'))

    if not label_file_paths:
        print("❌ No label files found.")
        return None

    print(f"✅ Found label file: {label_file_paths[0]}")
    bounding_boxes = Load_Predicated_ROI_Labels(label_file_paths[0])
    print("📦 Bounding box details:", bounding_boxes)

    return bounding_boxes


def bmd_predict(roi_image, features, bmd_model_path):
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    def mse(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)

    model = load_model(bmd_model_path, custom_objects={'mse': mse})
    return model.predict([roi_image, features]).flatten()


def T_score_predict(roi_image, features, model_path, json_path):
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    def mse(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)

    model = load_model(model_path, custom_objects={'mse': mse})

    with open(json_path, "r") as f:
        params = json.load(f)
    min_val, max_val = params["T_Score_min"], params["T_Score_max"]

    normalized = model.predict([roi_image, features]).flatten()
    return (normalized * (max_val - min_val)) + min_val


def Z_score_predict(roi_image, features, model_path, json_path):
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    def mse(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)

    model = load_model(model_path, custom_objects={'mse': mse})

    with open(json_path, "r") as f:
        params = json.load(f)
    min_val, max_val = params["Z_Score_min"], params["Z_Score_max"]

    normalized = model.predict([roi_image, features]).flatten()
    return (normalized * (max_val - min_val)) + min_val


def WHO_predict(roi_image, features, model_path):
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    def mse(y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)

    model = load_model(model_path, custom_objects={'mse': mse})
    probabilities = model.predict([roi_image, features])
    return np.argmax(probabilities)


def testing_models(predicted_bbox_output_file, yolo_model_path, bmd_model_path,
                   T_score_model_path, Z_score_model_path, WHO_model_path,
                   T_score_json_path, Z_score_json_path,
                   test_xray_image, test_features):

    bboxes = test_yolo(test_xray_image, yolo_model_path, predicted_bbox_output_file)
    cropped_image = create_cropped_image_test(test_xray_image, bboxes)

    bmd_indices = [0, 2, 3, 4, 5]
    z_score_indices = [0, 1, 3, 4, 5]
    t_who_indices = [0, 1, 4, 5]

    bmd_features = test_features[:, bmd_indices]
    z_score_features = test_features[:, z_score_indices]
    t_who_features = test_features[:, t_who_indices]

    bmd = bmd_predict(cropped_image, bmd_features, bmd_model_path)
    T_score = T_score_predict(cropped_image, t_who_features, T_score_model_path, T_score_json_path)
    Z_score = Z_score_predict(cropped_image, z_score_features, Z_score_model_path, Z_score_json_path)
    WHO_classification = WHO_predict(cropped_image, t_who_features, WHO_model_path)

    risk = ['NOT INCREASED', 'INCREASED', 'HIGH'][WHO_classification]

    return bmd, T_score, Z_score, WHO_classification, risk


def integration(file_path, features):
    model_dir = "models"
    config_dir = "config"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    download_model_if_needed("yolo.pt", f"{model_dir}/yolo.pt")
    download_model_if_needed("bmd_model.h5", f"{model_dir}/bmd_model.h5")
    download_model_if_needed("T_score_model.h5", f"{model_dir}/T_score_model.h5")
    download_model_if_needed("Z_score_model.h5", f"{model_dir}/Z_score_model.h5")
    download_model_if_needed("WHO_model.h5", f"{model_dir}/WHO_model.h5")
    download_model_if_needed("T_Score_normalization_params.json", f"{config_dir}/T_Score_normalization_params.json")
    download_model_if_needed("Z_Score_normalization_params.json", f"{config_dir}/Z_Score_normalization_params.json")

    return testing_models(
        predicted_bbox_output_file="static",
        yolo_model_path=f"{model_dir}/yolo.pt",
        bmd_model_path=f"{model_dir}/bmd_model.h5",
        T_score_model_path=f"{model_dir}/T_score_model.h5",
        Z_score_model_path=f"{model_dir}/Z_score_model.h5",
        WHO_model_path=f"{model_dir}/WHO_model.h5",
        T_score_json_path=f"{config_dir}/T_Score_normalization_params.json",
        Z_score_json_path=f"{config_dir}/Z_Score_normalization_params.json",
        test_xray_image=file_path,
        test_features=features
    )
