from ultralytics import YOLO
import cv2
import os
from PIL import Image, ImageDraw
import numpy as np
from utils import download_yolo11_face_detection

def download_yolo_model(yolo_model_path):
    # Download the YOLOv11 face detection model if it doesn't exist
    # model_path = "yolo11-face.pt"
    if not os.path.exists(yolo_model_path):
        download_yolo11_face_detection.download_yolo11_face_detection_model()
        assert os.path.exists(yolo_model_path), f"Model file {yolo_model_path} was not downloaded successfully"


def test_face_yolo11_detection():
    # Define paths
    models_dir = "./checkpoints"
    yolo_model_dir = os.path.join(models_dir, "yolo11_face_detection")
    yolo_model_path = os.path.join(yolo_model_dir, "model.pt")
    
    # Download model if it doesn't exist
    download_yolo_model(yolo_model_path)

    # Load the YOLOv11 face detection model
    model = YOLO(yolo_model_path)
    
    # Define image path
    image_path = "test/test_images/Elon_Musk.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist")
    
    # Load and convert image to RGB upfront
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference with minimal logging
    results = model(image_path, conf=0.25, iou=0.45, verbose=False)
    
    # Extract bounding boxes as NumPy array
    bounding_boxes = []
    cropped_faces = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Move to CPU and convert to NumPy
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Ensure valid crop region
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= image.shape[1] and y2 <= image.shape[0]:
                bounding_boxes.append([x1, y1, x2, y2])
                # Crop and convert to PIL Image
                cropped_face = image_rgb[y1:y2, x1:x2]
                if cropped_face.size > 0:
                    pil_image = Image.fromarray(cropped_face)
                    pil_image_resized = pil_image.resize((112, 112), Image.Resampling.BILINEAR)
                    cropped_faces.append(pil_image_resized)
    
    # Convert bounding boxes to NumPy array
    bounding_boxes_np = np.array(bounding_boxes, dtype=np.int32) if bounding_boxes else np.empty((0, 4), dtype=np.int32)
    
    # Print results
    print("Bounding Boxes (NumPy array):")
    print(bounding_boxes_np)
    
    print("\nCropped Faces (PIL Images):")
    for i, face in enumerate(cropped_faces):
        print(f"Face {i+1}: {face}")
    
    return bounding_boxes_np, cropped_faces
# if __name__ == "__main__":
test_face_yolo11_detection()