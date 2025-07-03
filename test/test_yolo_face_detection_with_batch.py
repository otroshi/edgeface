from ultralytics import YOLO
import cv2
import os
from PIL import Image
import numpy as np
import glob
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import download_yolo_face_detection

def download_yolo_model(yolo_model_path):
    """Download the YOLOv11 face detection model if it doesn't exist."""
    if not os.path.exists(yolo_model_path):
        download_yolo_face_detection.download_yolo_face_detection_model()

def face_yolo_detection(image_paths):
    """
    Perform face detection on a list of images using YOLOv11 model with batch inference.
    
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
    
    # Load and validate images
    valid_images = []
    valid_image_paths = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} does not exist. Skipping.")
            all_bounding_boxes.append(np.empty((0, 4), dtype=np.int32))
            all_cropped_faces.append([])
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image at {image_path}. Skipping.")
            all_bounding_boxes.append(np.empty((0, 4), dtype=np.int32))
            all_cropped_faces.append([])
            continue
        
        valid_images.append(image)
        valid_image_paths.append(image_path)
    
    # Perform batch inference if there are valid images
    if valid_images:
        # Convert images to RGB for processing
        images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in valid_images]
        
        # Run batch inference
        results = model.predict(source=valid_image_paths, conf=0.25, iou=0.45,
                                #  batch_size=16,
                                verbose=False)
        
        # Process results for each image
        for idx, result in enumerate(results):
            bounding_boxes = []
            cropped_faces = []
            image = valid_images[idx]  # Original BGR image for cropping
            image_rgb = images_rgb[idx]  # RGB image for PIL conversion
            
            # Extract bounding boxes
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
    
    # Ensure output lists match the input image_paths length
    # For skipped images, placeholders are already added during validation
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
    # Get all .jpg and .png files in a directory
    test_images_dir = "test/test_images"
    # Get all .jpg and .png files in the directory
    image_paths = (
        glob.glob(os.path.join(test_images_dir, "*.[jJ][pP][gG]")) +
        glob.glob(os.path.join(test_images_dir, "*.[pP][nN][gG]"))
    )
    import time
    t1 = time.time()
    all_bounding_boxes, all_cropped_faces = face_yolo_detection(image_paths)
    print("time: ", time.time() - t1)