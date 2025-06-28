from face_alignment.align import get_aligned_face


bbox, face = get_aligned_face("test/test_images/Elon_Musk.jpg", rgb_pil_image=None)
print("bbox, face: ", bbox, face)
