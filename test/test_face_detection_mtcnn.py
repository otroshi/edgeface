import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from face_alignment.align import get_aligned_face

bbox, face = get_aligned_face("test/test_images/Elon_Musk.jpg", rgb_pil_image=None)
print("type(bbox), bbox: ", type(bbox), bbox)
print("type(face), face: ", type(face), face)