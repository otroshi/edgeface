import sys
import os
from huggingface_hub import snapshot_download
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def download_yolo_face_detection_model():
    """
    Downloads the YOLOv11 face detection model from Hugging Face using snapshot_download.
    """
    print("Downloading YOLOv11 face detection model...")
    # Define the checkpoint directory
    checkpoint_dir = "./checkpoints"
    yolo_dir = os.path.join(checkpoint_dir, "yolo11_face_detection")
    # Create the checkpoint directory if it doesn't exist
    os.makedirs(yolo_dir, exist_ok=True)

    model_id = "AdamCodd/YOLOv11n-face-detection"
    
    try:
        # Download the model snapshot
        snapshot_download(
            repo_id=model_id,
            local_dir=yolo_dir,
            allow_patterns=["*.pt", "*.bin"],  # Only download model weight files
            local_dir_use_symlinks=False  # Avoid symlinks for clarity
        )
        print(f"Successfully downloaded model to {yolo_dir}")
    except Exception as e:
        print(f"Error during model download: {e}")
        sys.exit(1)
    finally:
        # Change back to the original directory
        os.chdir(os.path.abspath(os.path.join(yolo_dir, "..", "..")))

if __name__ == "__main__":
    download_yolo_face_detection_model()
    print("YOLOv11 face detection model download complete.")
