import cv2 
from utils import Fourier2, Fourier1
from typing import Optional
import numpy as np

class image:
    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(path)
        self.shape = self.image.shape
        self.height, self.width = self.image.shape[:2]
        self.channels = self.image.shape[2] if len(self.image.shape) > 2 else 1

    def noise(self, mean=0, std=0.1) -> np.ndarray:
        noise = np.random.normal(mean, std, self.image.shape)
        self.image = self.image + noise
        return self.image
    
    def normalize(self) -> np.ndarray: 
        self.image /= self.image.sum()
        return self.image
    
    def resize(self, width, height) -> np.ndarray:
        self.image = cv2.resize(self.image, (width, height))
        return self.image
    
    def f_12(self, second_image) -> np.ndarray:
        if second_image.sum != 1:
            second_image = second_image / second_image.sum()
        pfm = Fourier1(self.image, second_image)
        return pfm
    
    def f_22(self, second_image) -> np.ndarray:
        if second_image.sum != 1:
            second_image = second_image / second_image.sum()
        pfm = Fourier2(self.image, second_image)
        return pfm