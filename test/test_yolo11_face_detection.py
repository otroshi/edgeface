from ultralytics import YOLO
import cv2
import os
import json
from scripts import download_yolo11_face_detection
def download_yolo_model():
    # Download the YOLOv11 face detection model if it doesn't exist
    model_path = "yolo11-face.pt"
    if not os.path.exists(model_path):
        download_yolo11_face_detection.download_yolo11_face_detection_model()
        assert os.path.exists(model_path), f"Model file {model_path} was not downloaded successfully"

def test_face_yolo11_detection():
    models_dir = "./checkpoints"
    yolo_model_dir = os.path.join(models_dir, "yolo11_face_detection")
    yolo_model_path = os.path.join(yolo_model_dir, "model.pt")
    if not os.path.exists(yolo_model_path):
        download_yolo_model()

    # Load the YOLOv11 face detection model
    model = YOLO(yolo_model_path)
    image_path = "test/images/face.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist")
    results = model(image_path, conf=0.25, iou=0.45, show=True)
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
    # Process results and draw bounding boxes
    for result in results:
        boxes = result.boxes  # Get bounding boxes
        for box in boxes:
            # Extract coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            label = f"Face {confidence:.2f}"
            
            # Draw bounding box and label on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the image with bounding boxes
    cv2.imshow("YOLOv11 Face Detection", image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close the window

# if __name__ == "__main__":
test_face_yolo11_detection()