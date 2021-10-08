import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
import face_alignment

from params.params import path_to_chkpt
from dataset.video_extraction_conversion import generate_landmarks


path_to_model_weights = path_to_chkpt

vidpath = '/home/chern0g0r/workspace/keentools/prototype/Realistic-Neural-Talking-Head-Models/videos/{}/{}.mp4'
purepath = '/home/chern0g0r/workspace/keentools/prototype/Realistic-Neural-Talking-Head-Models/frames/{}/{}/pure/{}.jpg'
landpath = '/home/chern0g0r/workspace/keentools/prototype/Realistic-Neural-Talking-Head-Models/frames/{}/{}/land/{}.jpg'

videos = ['00001', '00161', '00176', '00239']
person = 'person3'


def get_image_landmarks(images):  # [image] -> landmarks, img
    face_aligner = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False, device='cuda:0')
    landmarks = generate_landmarks(images, face_aligner)
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
    for label in videos:
        # print(vidpath.format(person, label))
        # print(purepath.format(person, label, 0))
        # print(landpath.format(person, label, 0))
        cap = cv2.VideoCapture(vidpath.format(person, label))
        ret = True
        i = 0

        while ret:
            ret, frame = cap.read()

            if frame is not None:
                cv2.imwrite(purepath.format(person, label, i), frame)
                landmarks, img = get_image_landmarks([frame])
                # imgs.append(frame)

                cv2.imwrite(landpath.format(person, label, i), landmarks)

            i += 1
