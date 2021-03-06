import cv2
import numpy as np

from src.utils import rgb2gray


class Noiser:
    def __init__(self, s: float):
        self.s = s

    def noise_img(self, imPath: str) -> np.ndarray:
        """
        Given an image path, return a noisy image

        :param imgPath: the path to the image to be used as the base for the noise
        :type imgPath: str
        :return: The image with noise.
        """

        img = cv2.imread(imPath)  # img is a b&w image coded as a RGB one.
        gray_img = rgb2gray(img)
        noise = cv2.randn(np.zeros(gray_img.shape, float), (0), (self.s))

        return cv2.add(noise, gray_img, dtype=cv2.CV_8U)
