import cv2 
import sys
import os
sys.path.append('C:/Users/eriki/OneDrive/Documents/all_folder/Thesis/Thesis/utils')
from utils.utils import *
from typing import Optional
import numpy as np
import ot
import time

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
        noise_array = np.random.normal(mean, std, self.image.shape)
        self.image_noised = self.image + noise_array
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
    
    def split_image(self, noise_param=1e-3) -> np.ndarray:
        self.noise(std=noise_param)
        self.positive, self.negative = split_signed_measure(self.image_noised)
        return self.positive, self.negative

    def __repr__(self):
        return f"Image: {self.path}, Shape: {self.shape}"
    
    @staticmethod
    def process_images(image1, image2, noise_param):
        # Apply noise and split each image
        image1.split_image(noise_param=noise_param)
        image2.split_image(noise_param=noise_param)

        # Mix and normalize the positive and negative parts
        image1.image_post = image1.positive + image2.negative
        image2.image_post = image2.positive + image1.negative

        # Normalize the posterior images
        image1.image_post = image1.image_post / np.sum(image1.image_post)
        image2.image_post = image2.image_post / np.sum(image2.image_post)

        image1.image_noised = image1.image_noised / np.sum(image1.image_noised)
        image2.image_noised = image2.image_noised / np.sum(image2.image_noised)

    @classmethod
    def analyze_image_pair(cls, image1, image2, cost_matrix, num_samples, noise_param):
        w1_dists_noised = []
        times_w1 = []
        f_dists_noised = []
        times_f = []
        l2_dists_noised = []
        times_l2 = []

        for _ in range(num_samples):
            cls.process_images(image1, image2, noise_param)  # Adjusted for class method context

            w_dist, w_time = calculate_and_time_wasserstein(image1.image_post, image2.image_post, cost_matrix)
            f_dist, f_time = calculate_and_time_fourier(image1.image_post, image2.image_post)
            l_dist, l_time = calculate_and_time_l2(image1.image_noised, image2.image_noised)

            w1_dists_noised.append(w_dist)
            times_w1.append(w_time)
            f_dists_noised.append(f_dist)
            times_f.append(f_time)
            l2_dists_noised.append(l_dist)
            times_l2.append(l_time)

        w1_dist_noised = np.mean(w1_dists_noised)
        f_dist_noised = np.mean(f_dists_noised)
        l2_dist_noised = np.mean(l2_dists_noised)

        time_w1 = np.mean(times_w1)
        time_f = np.mean(times_f)
        time_l2 = np.mean(times_l2)

        return w1_dist_noised, f_dist_noised, l2_dist_noised, time_w1, time_f, time_l2