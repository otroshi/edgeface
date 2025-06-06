import sys
import os
import numpy as np
from face_alignment import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime
mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(112, 112))

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_aligned_face(image_input, rgb_pil_image=None):
    """Get aligned face from various input types.
    
    Args:
        image_input: Can be one of:
            - str: Path to image file
            - np.ndarray: RGB image array of shape (H,W,3)
            - None: When using rgb_pil_image parameter
        rgb_pil_image (PIL.Image, optional): RGB PIL Image
        
    Returns:
        PIL.Image: Aligned face or None if detection fails
    """
    if rgb_pil_image is not None:
        assert isinstance(rgb_pil_image, Image.Image), 'rgb_pil_image must be a PIL Image'
        img = rgb_pil_image
    elif isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        assert len(image_input.shape) == 3 and image_input.shape[2] == 3, \
            'NumPy array must be RGB with shape (H,W,3)'
        img = Image.fromarray(image_input)
    else:
        raise TypeError("Input must be file path, numpy array, or PIL Image")

    # find face
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1)
        face = faces[0] if faces else None
    except Exception as e:
        print('Face detection Failed due to error.')
        print(e)
        face = None

    return face


