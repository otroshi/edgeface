from ultralytics import YOLO
import cv2
import os
from PIL import Image
import numpy as np
import glob
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import download_yolo_face_detection

def initialize_yolo_model(yolo_model_path):
    """Initialize YOLO model."""
    if not os.path.exists(yolo_model_path):
        download_yolo_face_detection.download_yolo_face_detection_model()
    return YOLO(yolo_model_path)

def process_image_results(image, image_rgb, boxes):
    """Process bounding boxes and crop faces for a single image."""
    bounding_boxes, cropped_faces = [], []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= image.shape[1] and y2 <= image.shape[0]:
            bounding_boxes.append([x1, y1, x2, y2])
            cropped_face = image_rgb[y1:y2, x1:x2]
            if cropped_face.size > 0:
                pil_image = Image.fromarray(cropped_face).resize((112, 112), Image.Resampling.BILINEAR)
                cropped_faces.append(pil_image)
    return np.array(bounding_boxes, dtype=np.int32) if bounding_boxes else np.empty((0, 4), dtype=np.int32), cropped_faces

def process_batch(model, image_paths, all_bounding_boxes, all_cropped_faces):
    """Process images in batch mode using list comprehensions for efficiency."""
    # Validate and load images, filter out invalid ones
    valid_data = [(cv2.imread(path), path) for path in image_paths if os.path.exists(path)]
    valid_images, valid_image_paths = zip(*[(img, path) for img, path in valid_data if img is not None]) if valid_data else ([], [])

    # Append empty results for invalid images
    for path in image_paths:
        if not os.path.exists(path) or cv2.imread(path) is None:
            all_bounding_boxes.append(np.empty((0, 4), dtype=np.int32))
            all_cropped_faces.append([])
            print(f"Warning: {'not found' if not os.path.exists(path) else 'failed to load'} {path}. Skipping.")

    # Process valid images
    if valid_images:
        images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in valid_images]
        results = model.predict(source=valid_image_paths, conf=0.25, iou=0.45, verbose=False)
        # Process results with comprehension
        [all_bounding_boxes.append(bboxes) and all_cropped_faces.append(faces) 
         for bboxes, faces in [process_image_results(img, rgb, result.boxes.xyxy.cpu().numpy()) 
                               for img, rgb, result in zip(valid_images, images_rgb, results)]]

def process_individual(model, image_paths, all_bounding_boxes, all_cropped_faces):
    """Process images individually."""
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found. Skipping.")
            all_bounding_boxes.append(np.empty((0, 4), dtype=np.int32))
            all_cropped_faces.append([])
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load {image_path}. Skipping.")
            all_bounding_boxes.append(np.empty((0, 4), dtype=np.int32))
            all_cropped_faces.append([])
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image_path, conf=0.25, iou=0.45, verbose=False)
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            bboxes, faces = process_image_results(image, image_rgb, boxes)
            all_bounding_boxes.append(bboxes)
            all_cropped_faces.append(faces)

def face_yolo_detection(yolo_model_path, image_paths, use_batch=True):
    """Perform face detection using YOLOv11 with batch or individual processing."""
    model = initialize_yolo_model(yolo_model_path)
    all_bounding_boxes, all_cropped_faces = [], []
    
    if use_batch:
        process_batch(model, image_paths, all_bounding_boxes, all_cropped_faces)
    else:
        process_individual(model, image_paths, all_bounding_boxes, all_cropped_faces)
    
    # for i, (bboxes, faces) in enumerate(zip(all_bounding_boxes, all_cropped_faces)):
    #     print(f"\nImage {i+1} ({image_paths[i]}):")
    #     print("Bounding Boxes:", bboxes)
    #     print("Cropped Faces:", [f"Face {j+1}: {face}" for j, face in enumerate(faces)])
    
    return all_bounding_boxes, all_cropped_faces

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 face detection")
    parser.add_argument("--use-batch", action="store_true", default=True, help="Use batch processing (default: True)")
    parser.add_argument("--image-dir", type=str, default="test/test_images", help="Input image directory")
    parser.add_argument("--yolo-model-path", type=str, default="checkpoints/yolo11_face_detection/model.pt", help="YOLO model path")
    
    args = parser.parse_args()
    
    image_paths = (glob.glob(os.path.join(args.image_dir, "*.[jJ][pP][gG]")) + 
                   glob.glob(os.path.join(args.image_dir, "*.[pP][nN][gG]"))
                )
    
    if args.yolo_model_path:
        yolo_model_path = args.yolo_model_path
    else:
        yolo_model_path = os.path.join("checkpoints", "yolo11_face_detection", "model.pt")

    # import time
    # t1 = time.time()
    # all_bounding_boxes, all_cropped_faces = face_yolo_detection(yolo_model_path, image_paths, args.use_batch)
    # print("Time taken:", time.time() - t1)

    import time
    num_runs = 50
    batch_times, individual_times = [], []
    
    # Benchmark batch processing
    for _ in range(num_runs):
        t1 = time.time()
        face_yolo_detection(yolo_model_path, image_paths, use_batch=True)
        batch_times.append(time.time() - t1)
    
    # Benchmark individual processing
    for _ in range(num_runs):
        t1 = time.time()
        face_yolo_detection(yolo_model_path, image_paths, use_batch=False)
        individual_times.append(time.time() - t1)
    
    # Calculate and print average times
    avg_batch_time = sum(batch_times) / num_runs
    avg_individual_time = sum(individual_times) / num_runs
    
    print(f"\nBenchmark Results (over {num_runs} runs):")
    print(f"Average Batch Processing Time: {avg_batch_time:.4f} seconds")
    print(f"Average Individual Processing Time: {avg_individual_time:.4f} seconds")