import sys
import os
import torch
from . import mtcnn
from .face_yolo import face_yolo_detection
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

mtcnn_model = mtcnn.MTCNN(device=DEVICE, crop_size=(112, 112))

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def handle_image_mtcnn(img_path, pil_img):
        img = Image.open(img_path).convert('RGB') if pil_img is None else pil_img
        assert isinstance(img, Image.Image), 'Face alignment requires PIL image or path'
        try:
            bboxes, faces = mtcnn_model.align_multi(img, limit=1)
            return bboxes[0], faces[0]
        except Exception as e:
            print(f'Face detection failed: {e}')
            return None, None

def get_aligned_face(image_path_or_image_paths, rgb_pil_image=None, algorithm='mtcnn'):
    if algorithm=='mtcnn':
        if isinstance(image_path_or_image_paths, list):
            results = [handle_image_mtcnn(path, rgb_pil_image) for path in image_path_or_image_paths]
            return results
        elif isinstance(image_path_or_image_paths, str):
            return [handle_image_mtcnn(image_path_or_image_paths, rgb_pil_image)]
        else:
            raise TypeError("image_path_or_image_paths must be a list or string") 

    elif algorithm=='yolo':
        if isinstance(image_path_or_image_paths, list):
            image_paths = image_path_or_image_paths
            results = face_yolo_detection(image_paths,
                        # yolo_model_path="checkpoints/yolo11_face_detection/model.pt",
                        use_batch=True)
        elif isinstance(image_path_or_image_paths, str):
            image_paths = [image_path_or_image_paths]
            results = face_yolo_detection(image_paths,
                        # yolo_model_path="checkpoints/yolo11_face_detection/model.pt",
                        use_batch=True)
        else:
            raise TypeError("image_path_or_image_paths must be a list or string") 
        results = list(results)
    return results
