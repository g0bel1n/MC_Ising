import itertools
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils import CliqueSum


class Sampler(ABC):
    # Can be Metropolis or Gibbs

    def __init__(self, alpha: float, beta: float, sigma: float):
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

    @abstractmethod
    def U(self, new_value: int, pixel: tuple[int, int], img: np.ndarray) -> float:
        pass

    def getProbas(self, pixel: tuple[int, int], img: np.ndarray) -> tuple[float, float]:
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
            np.exp(self.U(0, pixel, img) - (img[pixel] ** 2) / (2 * self.sigma)),
            np.exp(
                self.U(255, pixel, img) - ((img[pixel] - 255) ** 2) / (2 * self.sigma)
            ),
        ]
        return probas / np.sum(probas)

    @abstractmethod
    def sample(self, img: np.ndarray) -> np.ndarray:
        pass


class GibbsSampler(Sampler):
    def __init__(
        self, alpha: float, beta: float, sigma: float, burn_in: int, n_samples: int
    ):
        super().__init__(alpha, beta, sigma)
        self.n_samples = n_samples
        self.burn_in = burn_in

    def U(self, new_value: int, ind_pixel: tuple[int, int], img: np.ndarray) -> float:
        """
        > The function `U` takes in a new value, an index of a pixel, and an image, and returns the sum
        of the new value multiplied by the alpha parameter, and the sum of the cliques multiplied by the
        beta parameter

        :param new_value: the new value of the pixel
        :type new_value: int
        :param ind_pixel: the index of the pixel we're trying to find the energy of
        :type ind_pixel: tuple[int, int]
        :param img: the image we're trying to denoise
        :type img: np.ndarray
        :return: The energy of the pixel at the given index.
        """
        return self.alpha * new_value + self.beta * CliqueSum(
            new_value=new_value, ind_pixel=ind_pixel, img=img
        )

    def sample(self, img: np.ndarray, gif: bool = False) -> np.ndarray:

        """
        We start with a random image, and then we iterate over all the pixels, and for each pixel we
        sample a new value from the conditional distribution of that pixel given the rest of the image

        :param img: the image to be denoised
        :type img: np.ndarray
        :param n_samples: The number of samples to take, defaults to 20
        :type n_samples: int (optional)
        :param burn_in: The number of iterations to run before starting to sample, defaults to 20
        :type burn_in: int (optional)
        :return: The average of the samples.
        """

        L, W = img.shape
        X = img.copy()

        if gif:
            i: int = 0
            plt.imsave(f"data/output/gif/{i}.png", X, cmap="gray")
            i += 1
        with tqdm(total=self.burn_in * L * W, desc="Burn in", ascii="░▒█") as bbar:
            for _ in range(self.burn_in):
                for l, w in itertools.product(range(L), range(W)):
                    probas = self.getProbas(pixel=(l, w), img=X)
                    X[l, w] = np.random.choice((0, 255), 1, p=probas)
                    bbar.update(1)
                if gif:
                    plt.imsave(f"data/output/gif/{i}.png", X, cmap="gray")
                    i += 1

        avg = np.zeros_like(img).astype(np.uint64)
        with tqdm(total=self.n_samples * L * W, desc="Sampling", ascii="░▒█") as pbar:
            for _ in range(self.n_samples):
                for l, w in itertools.product(range(L), range(W)):
                    probas = self.getProbas(pixel=(l, w), img=X)
                    X[l, w] = np.random.choice((0, 255), 1, p=probas)
                    avg += X
                    pbar.update(1)
                if gif:
                    plt.imsave(f"data/output/gif/{i}.png", X, cmap="gray")
                    i += 1

        avg = avg.astype(float)
        avg = avg / (L * W * self.n_samples)
        avg[avg >= 255.0 / 2] = 255
        avg[avg < 255.0 / 2] = 0
        avg = avg.astype(np.uint8)
        return avg
