import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
import face_alignment

from params.params import path_to_chkpt
from dataset.video_extraction_conversion import generate_landmarks, select_frames


path_to_model_weights = path_to_chkpt

impath = '/home/chern0g0r/workspace/keentools/prototype/Realistic-Neural-Talking-Head-Models/examples/fine_tuning/test_images/test_cranston.jpeg'
vidpath = '/home/chern0g0r/workspace/keentools/prototype/Realistic-Neural-Talking-Head-Models/examples/fine_tuning/test_video.mp4'


def get_image_landmarks(image):  # -> landmarks, img
    face_aligner = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False, device='cuda:0')
    landmarks = generate_landmarks(image, face_aligner)
    landmarks = torch.from_numpy(np.array(landmarks)).type(
        dtype=torch.float)
    landmarks = landmarks.transpose(2, 4)/255

    g_idx = torch.randint(low=0, high=1, size=(1, 1))
    img = landmarks[g_idx, 0].squeeze()*255
    frame_marks = landmarks[g_idx, 1].squeeze()*255
    img = np.transpose(img.numpy(), (2, 1, 0))  # _, _, 3
    frame_marks = np.transpose(frame_marks.numpy(), (2, 1, 0))  # _, _, 3
    return frame_marks, img


def change_landmarks(landmarks):  # -> landmarks
    pass


def test():
    img = cv2.imread(impath)
    landmarks, img = get_image_landmarks([img])
    cv2.imwrite('x.jpg', img)
    cv2.imwrite('gy.jpg', landmarks)
