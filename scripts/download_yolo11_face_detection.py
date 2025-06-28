import subprocess
import os

def download_yolo11_face_detection_model():
    """
    Downloads the YOLOv11 face detection model from Hugging Face.
    """
    print("Downloading YOLOv11 face detection model...")
    # Define the checkpoint directory
    checkpoint_dir = "./checkpoints"
    yolo_dir = os.path.join(checkpoint_dir, "yolo11_face_detection")
    # Create the checkpoint directory if it doesn't exist
    os.makedirs(yolo_dir, exist_ok=True)

    model_id = "AdamCodd/YOLOv11n-face-detection"
    platform = "https://huggingface.co"
    # URL of the Hugging Face repository
    repo_url = os.path.join(platform, model_id)

    # Run the git clone command
    try:
        subprocess.run(["git", "clone", repo_url, yolo_dir], check=True)
        print(f"Successfully cloned repository to {yolo_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")

if __name__ == "__main__":
    download_yolo11_face_detection_model()
    print("YOLOv11 face detection model download complete.")