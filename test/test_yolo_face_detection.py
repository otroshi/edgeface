from ultralytics import YOLO
import cv2
import os
from PIL import Image
import numpy as np
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import download_yolo_face_detection

def download_yolo_model(yolo_model_path):
    """Download the YOLOv11 face detection model if it doesn't exist."""
    # if not os.path.exists(yolo_model_path):
    download_yolo_face_detection.download_yolo_face_detection_model()


def face_yolo_detection(image_paths):
    """
    Perform face detection on a list of images using YOLOv11 model.
    
    Args:
        image_paths (list): List of file paths to the input images.
        
    Returns:
        tuple: 
            - list of np.ndarray: List of bounding boxes for each image, where each array has shape (N, 4) for N faces.
            - list of list: List of cropped face PIL Images for each image.
    """
    # Define paths
    models_dir = "./checkpoints"
    yolo_model_dir = os.path.join(models_dir, "yolo11_face_detection")
    yolo_model_path = os.path.join(yolo_model_dir, "model.pt")

    # Download model if it doesn't exist
    if not os.path.exists(yolo_model_path):
        download_yolo_model(yolo_model_path)

    # Load the YOLOv11 face detection model
    model = YOLO(yolo_model_path)
    
    # Initialize lists to store results for all images
    all_bounding_boxes = []
    all_cropped_faces = []
    
    # Process each image
    for image_path in image_paths:
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} does not exist. Skipping.")
            all_bounding_boxes.append(np.empty((0, 4), dtype=np.int32))
            all_cropped_faces.append([])
            continue
        
        # Load and convert image to RGB
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image at {image_path}. Skipping.")
            all_bounding_boxes.append(np.empty((0, 4), dtype=np.int32))
            all_cropped_faces.append([])
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference with minimal logging
        results = model(image_path, conf=0.25, iou=0.45, verbose=False)
        
        # Extract bounding boxes and cropped faces for the current image
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
        
        # Convert bounding boxes to NumPy array for the current image
        bounding_boxes_np = np.array(bounding_boxes, dtype=np.int32) if bounding_boxes else np.empty((0, 4), dtype=np.int32)
        
        # Append results for the current image
        all_bounding_boxes.append(bounding_boxes_np)
        all_cropped_faces.append(cropped_faces)
    
    # Print results for all images
    for i, (bboxes, faces) in enumerate(zip(all_bounding_boxes, all_cropped_faces)):
        print(f"\nResults for Image {i+1} ({image_paths[i]}):")
        print("Bounding Boxes (NumPy array):")
        print(bboxes)
        print("\nCropped Faces (PIL Images):")
        for j, face in enumerate(faces):
            print(f"Face {j+1}: {face}")
    
    return all_bounding_boxes, all_cropped_faces

# Example usage
if __name__ == "__main__":
    image_paths = [
        "test/test_images/Elon_Musk.jpg",
        "test/test_images/Gal Gado.jpg",  # Replace with actual image paths
    ]
    all_bounding_boxes, all_cropped_faces = face_yolo_detection(image_paths)