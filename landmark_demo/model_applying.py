import torch
import cv2
from matplotlib import pyplot as plt

from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import Generator
from webcam_demo.webcam_extraction_conversion import *

from params.params import path_to_chkpt
from tqdm import tqdm


class ModelApplyer:
    def __init__(self,
                 path_to_model_weights='finetuned_model.tar',
                 path_to_embedding='e_hat_video.tar',
                 path_to_mp4='examples/fine_tuning/test_video.mp4'):

        self.device = torch.device("cuda:0")
        self.cpu = torch.device("cpu")
        self.device = self.cpu

        checkpoint = torch.load(path_to_model_weights, map_location=self.cpu)
        self.e_hat = torch.load(path_to_embedding, map_location=self.cpu)
        self.e_hat = self.e_hat['e_hat'].to(self.device)

        self.G = Generator(256, finetuning=True, e_finetuning=self.e_hat)
        self.G.eval()

        """Training Init"""
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.G.to(self.device)

    def apply(self, landmark):
        with torch.no_grad():
            landmark = torch.from_numpy(landmark).type(torch.FloatTensor)
            landmark = landmark.transpose(0, 2)
            landmark = landmark.unsqueeze(0)

            x_hat = self.G(landmark, self.e_hat)

            out1 = x_hat.transpose(1, 3)[0]
            out1 = out1.to(self.cpu).numpy()

            fake = cv2.cvtColor(out1*255, cv2.COLOR_BGR2RGB)

        return fake
