import cv2 
import sys
import os
sys.path.append('C:/Users/eriki/OneDrive/Documents/all_folder/Thesis/Thesis/utils')
from utils.utils import Fourier2, Fourier1
from typing import Optional
import numpy as np

class Image:
    def __init__(self, resolution: int, category: str, index: int, 
                 main_path: str = 'C:/Users/eriki/OneDrive/Documents/all_folder/Thesis/DOTMark_1.0/Pictures'):
        self.resolution = resolution
        self.category = category
        if len(index.__str__()) == 1:
            index = f"0{index+1}"  # The index is between 0 and 9, so we add 1 to it
        self.index = index.__str__()
        self.path = os.path.join(main_path, category, f"picture{resolution}_10{index}.png")
        self.image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        self.shape = self.image.shape
        if self.shape[0] != self.resolution:
            self.image = self.resize(self.resolution, self.resolution)
            self.shape = self.image.shape
        self.normalize()


    def noise(self, mean=0, std=0.1) -> np.ndarray:
        self.noise = np.random.normal(mean, std, self.image.shape)
        self.image_noised = self.image + self.noise
        return self.image_noised
    
    def normalize(self) -> np.ndarray: 
        self.image = self.image / self.image.sum()
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
    
    def split_image(self) -> np.ndarray:
        self.noise()
        self.positive = self.image[self.image > 0]
        self.negative = self.image[self.image < 0]
        return self.positive, self.negative

    def __repr__(self):
        return f"Image: {self.path}, Shape: {self.shape}, Channels: {self.channels}"
    