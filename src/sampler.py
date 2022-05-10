from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Sampler(ABC):
    # Can be Metropolis or Gibbs

    def __init__(
        self, alpha: Optional[float], beta: Optional[float], tau_square: Optional[float]
    ):

        self.alpha = alpha
        self.beta = beta
        self.tau_square = tau_square

    @abstractmethod
    def U(
        self,
        new_value: int,
        pixel: tuple[int, int],
        img: np.ndarray,
        alpha: float,
        beta: float,
    ) -> float:
        pass

    def getProbas(
        self,
        pixel: tuple[int, int],
        img: np.ndarray,
        alpha: float,
        beta: float,
        tau_square: float,
    ) -> tuple[float, float]:
        """
        > The function takes in a pixel and an image and returns the probability of the pixel being
        black and white

        :param pixel: the pixel we're looking at
        :type pixel: tuple[int, int]
        :param img: the image we're working on
        :type img: np.ndarray
        :return: The probability of the pixel being black or white.
        """
        probas = [
            np.exp(
                self.U(0, pixel, img, alpha, beta)
                - (img[pixel] ** 2) / (2 * tau_square)
            ),
            np.exp(
                self.U(1, pixel, img, alpha, beta)
                - ((img[pixel] - 1) ** 2) / (2 * tau_square)
            ),
        ]
        return probas / np.sum(probas)

    @abstractmethod
    def sample(self, img: np.ndarray) -> np.ndarray:
        pass
