import numpy as np
import torch
from torchvision import transforms
from backbones import get_model
from face_alignment import align
from .distance_utils import compute_distance, get_verification_threshold

class Verifier():
    def __init__(self, model_name: str = "edgeface_base"):
        self.model_name = model_name
        self.checkpoint_path=f'checkpoints/{self.model_name}.pt'
        self.model = get_model(self.model_name)
        self.model.load_state_dict(torch.load(self.checkpoint_path, 
                                              map_location='cpu'))
        
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        aligned_face = align.get_aligned_face(image)
        input_tensor = self.preprocess(aligned_face).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(input_tensor).squeeze().numpy()
        return embedding
    
    def verify(self, img_1: np.ndarray, img_2: np.ndarray):
        img_1 = align.get_aligned_face(img_1)
        img_2 = align.get_aligned_face(img_2)
        transformed_input_1 = self.transform(img_1).unsqueeze(0)
        transformed_input_2 = self.transform(img_2).unsqueeze(0)

        with torch.no_grad():
            embedding_1 = self.model(transformed_input_1).squeeze().numpy()
            embedding_2 = self.model(transformed_input_2).squeeze().numpy()

        print(embedding_1.shape, embedding_2.shape)

    def verify(self, image1: np.ndarray, image2: np.ndarray, dist_metric: str = "cosine") -> bool:
        embedding1 = self.extract_embedding(image1)
        embedding2 = self.extract_embedding(image2)

        distance = compute_distance(embedding1, embedding2, dist_metric)
        threshold = get_verification_threshold(dist_metric)
        is_same_person = distance <= threshold

        print(f"Distance ({dist_metric}): {distance:.6f} | Threshold: {threshold} | Match: {is_same_person}")
        return distance, threshold, is_same_person