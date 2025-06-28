from ultralytics import YOLO
import cv2
import os
import json
from scripts import download_yolo11_face_detection

def download_yolo_model(yolo_model_path):
    # Download the YOLOv11 face detection model if it doesn't exist
    # model_path = "yolo11-face.pt"
    if not os.path.exists(yolo_model_path):
        download_yolo11_face_detection.download_yolo11_face_detection_model()
        assert os.path.exists(yolo_model_path), f"Model file {yolo_model_path} was not downloaded successfully"

def test_face_yolo11_detection():
    models_dir = "./checkpoints"
    yolo_model_dir = os.path.join(models_dir, "yolo11_face_detection")
    yolo_model_path = os.path.join(yolo_model_dir, "model.pt")
    if not os.path.exists(yolo_model_path):
        download_yolo_model(yolo_model_path)

    # Load the YOLOv11 face detection model
    model = YOLO(yolo_model_path)
    image_path = "test/test_images/Elon_Musk.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist")
    results = model(image_path, conf=0.25, iou=0.45, show=True)
    
    # print(results)
# if __name__ == "__main__":
test_face_yolo11_detection()