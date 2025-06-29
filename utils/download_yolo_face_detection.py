import subprocess
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def download_yolo_face_detection_model():
    """
    Downloads the YOLOv11 face detection model from Hugging Face using Git LFS.
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

    # Check if Git LFS is installed
    try:
        subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
        print("Git LFS is installed.")
    except subprocess.CalledProcessError:
        print("Git LFS is not installed. Please install Git LFS and try again.")
        sys.exit(1)

    # Initialize Git LFS in the repository after cloning
    try:
        # Clone the repository
        subprocess.run(["git", "clone", repo_url, yolo_dir], check=True)
        print(f"Successfully cloned repository to {yolo_dir}")

        # Change to the cloned repository directory
        os.chdir(yolo_dir)
        
        # Initialize Git LFS
        subprocess.run(["git", "lfs", "install"], check=True)
        print("Git LFS initialized.")

        # Track large files (e.g., model weights like .pt or .bin files)
        subprocess.run(["git", "lfs", "track", "*.pt"], check=True)
        subprocess.run(["git", "lfs", "track", "*.bin"], check=True)
        print("Tracked large files with Git LFS.")

        # Pull LFS-tracked files
        subprocess.run(["git", "lfs", "pull"], check=True)
        print("Successfully pulled large files with Git LFS.")
    except subprocess.CalledProcessError as e:
        print(f"Error during repository cloning or Git LFS setup: {e}")
        sys.exit(1)
    finally:
        # Change back to the original directory
        os.chdir(os.path.abspath(os.path.join(yolo_dir, "..", "..")))

if __name__ == "__main__":
    download_yolo_face_detection_model()
    print("YOLOv11 face detection model download complete.")