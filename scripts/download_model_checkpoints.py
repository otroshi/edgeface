from huggingface_hub import snapshot_download

# List of model IDs
model_id_list = [
    "Idiap/EdgeFace-Base",
    "Idiap/EdgeFace-XS-GAMMA",
    "Idiap/EdgeFace-XXS",
    "Idiap/EdgeFace-S-GAMMA"
]

# Directory to save the models
cache_dir = "./checkpoints"

# Download each model to the specified cache directory
for model_id in model_id_list:
    try:
        print(f"Downloading {model_id}...")
        snapshot_download(repo_id=model_id, local_dir=cache_dir)
        print(f"Successfully downloaded {model_id} to {cache_dir}")
    except Exception as e:
        print(f"Error downloading {model_id}: {str(e)}")
