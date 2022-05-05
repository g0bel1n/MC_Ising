from scipy.stats import invgamma
import logging
from numpy.random import default_rng
from src.sampler import Sampler
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from .utils import CliqueSum


class GibbsSampler(Sampler):
    def __init__(
        self, alpha: float, beta: float, tau_square: float, burn_in: int, n_samples: int
    ):
        super().__init__(alpha, beta, tau_square)
        self.n_samples = n_samples
        self.burn_in = burn_in

    def U(
        self,
        new_value: int,
        ind_pixel: tuple[int, int],
        img: np.ndarray,
        alpha: float,
        beta: float,
    ) -> float:
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
        return alpha * new_value + beta * CliqueSum(
            new_value=new_value, ind_pixel=ind_pixel, img=img
        )

    def sample(self, img: np.ndarray, gif: bool = False) -> dict:

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

        changes = []
        L, W = img.shape
        X = img.copy() / 255.0
        print(f"{np.sum(X)}")
        tau_square = self.tau_square
        tau_square_storage = [0.0] if self.tau_square is None else [self.tau_square]

        alpha_invgamma, beta_invgamma = 2, 1
        mu_alpha, mu_beta = 0, 1
        sigma_square_alpha, sigma_square_beta = 1.0 / 2, 1.0 / 2

        alpha = (
            np.random.normal(mu_alpha, sigma_square_alpha)
            if self.alpha is None
            else self.alpha
        )
        beta = (
            np.random.normal(mu_beta, sigma_square_beta)
            if self.beta is None
            else self.beta
        )

        alpha_storage = [alpha]
        beta_storage = [beta]

        prop_to_posterior_alpha = np.exp(
            alpha * np.sum(X) + ((alpha - mu_alpha) ** 2) / (2 * sigma_square_alpha)
        )
        prop_to_posterior_beta = np.exp(
            beta
            * np.sum(
                [
                    CliqueSum(new_value=int(X[l, w]), ind_pixel=(l, w), img=X)
                    for l, w in itertools.product(range(L), range(W))
                ]
            )
            + ((beta - mu_beta) ** 2) / (2 * sigma_square_beta)
        )

        if gif:
            i: int = 0
            plt.imsave(f"data/output/gif/{i}.png", X, cmap="gray")
            i += 1
        with tqdm(total=self.burn_in * L * W, desc="Burn in", ascii="░▒█") as bbar:
            for _ in range(self.burn_in):
                if self.tau_square is None:
                    a, b = (
                        alpha_invgamma + (L * W) / 2,
                        beta_invgamma + np.sum(np.square(img / 255.0 - X)) / 2,
                    )
                    tau_square = invgamma(a=a, scale=b).rvs()

                if self.alpha or self.beta is None:
                    new_alpha_candidate, new_beta_candidate = (
                        np.random.normal(mu_alpha, sigma_square_alpha),
                        np.random.normal(mu_beta, sigma_square_beta),
                    )
                    new_prop_to_posterior_alpha = np.exp(
                        new_alpha_candidate * np.sum(X)
                        + ((new_alpha_candidate - mu_alpha) ** 2)
                        / (2 * sigma_square_alpha)
                    )
                    new_prop_to_posterior_beta = np.exp(
                        new_beta_candidate
                        * np.sum(
                            [
                                CliqueSum(
                                    new_value=int(X[l, w]), ind_pixel=(l, w), img=X
                                )
                                for l, w in itertools.product(range(L), range(W))
                            ]
                        )
                        + ((new_beta_candidate - mu_beta) ** 2)
                        / (2 * sigma_square_beta)
                    )
                    if (
                        min(1, new_prop_to_posterior_alpha / prop_to_posterior_alpha)
                        <= np.random.uniform()
                    ):
                        prop_to_posterior_alpha = new_prop_to_posterior_alpha
                        alpha = new_alpha_candidate
                    if (
                        min(1, new_prop_to_posterior_beta / prop_to_posterior_beta)
                        <= np.random.uniform()
                    ):
                        prop_to_posterior_beta = new_prop_to_posterior_beta
                        beta = new_beta_candidate

                change = 0
                alpha_storage.append(alpha)
                beta_storage.append(beta)

                for l, w in itertools.product(range(L), range(W)):
                    probas = self.getProbas(
                        pixel=(l, w),
                        img=X,
                        alpha=alpha,
                        beta=beta,
                        tau_square=tau_square,
                    )
                    new_x = np.random.choice((0, 1), 1, p=probas)
                    if new_x != X[l, w]:
                        change += 1
                    X[l, w] = new_x
                    bbar.update(1)
                    if gif and l % 10 == 0 and w == 0:
                        plt.imsave(f"data/output/gif/{i}.png", X, cmap="gray")
                        i += 1
                changes.append(change)

        avg = np.zeros_like(img).astype(np.float64)
        print(img / 255.0 - X)
        with tqdm(total=self.n_samples * L * W, desc="Sampling", ascii="░▒█") as pbar:
            for _ in range(self.n_samples):

                if self.tau_square is None:
                    a, b = (
                        alpha_invgamma + (L * W) / 2,
                        beta_invgamma + np.sum(np.square(img / 255.0 - X)) / 2,
                    )
                    tau_square = invgamma(a=a, scale=b).rvs()

                if self.alpha or self.beta is None:
                    new_alpha_candidate, new_beta_candidate = (
                        np.random.normal(mu_alpha, sigma_square_alpha),
                        np.random.normal(mu_beta, sigma_square_beta),
                    )
                    new_prop_to_posterior_alpha = np.exp(
                        new_alpha_candidate * np.sum(X)
                        + ((new_alpha_candidate - mu_alpha) ** 2)
                        / (2 * sigma_square_alpha)
                    )
                    new_prop_to_posterior_beta = np.exp(
                        new_beta_candidate
                        * np.sum(
                            [
                                CliqueSum(
                                    new_value=int(X[l, w]), ind_pixel=(l, w), img=X
                                )
                                for l, w in itertools.product(range(L), range(W))
                            ]
                        )
                        + ((new_beta_candidate - mu_beta) ** 2)
                        / (2 * sigma_square_beta)
                    )
                    if (
                        min(1, new_prop_to_posterior_alpha / prop_to_posterior_alpha)
                        <= np.random.uniform()
                    ):
                        prop_to_posterior_alpha = new_prop_to_posterior_alpha
                        alpha = new_alpha_candidate
                    if (
                        min(1, new_prop_to_posterior_beta / prop_to_posterior_beta)
                        <= np.random.uniform()
                    ):
                        prop_to_posterior_beta = new_prop_to_posterior_beta
                        beta = new_beta_candidate

                tau_square_storage.append(tau_square)
                alpha_storage.append(alpha)
                beta_storage.append(beta)

                change = 0

                for l, w in itertools.product(range(L), range(W)):

                    probas = self.getProbas(
                        pixel=(l, w),
                        img=X,
                        alpha=alpha,
                        beta=beta,
                        tau_square=tau_square,
                    )
                    new_x = np.random.choice((0, 1), 1, p=probas)
                    if new_x != X[l, w]:
                        change += 1
                    X[l, w] = new_x
                    avg += X
                    pbar.update(1)
                    if gif and l % 20 == 0 and w == 0:
                        plt.imsave(f"data/output/gif/{i}.png", X, cmap="gray")
                        i += 1
                changes.append(change)

        avg = avg.astype(float)
        avg = avg / (L * W * self.n_samples)
        avg[avg >= 1.0 / 2] = 255
        avg[avg < 1.0 / 2] = 0
        avg = avg.astype(np.uint8)

        plt.plot([np.log(change) if change > 0 else 0 for change in changes])
        plt.show()
        plt.savefig("changes")
        return {
            "img": avg,
            "alpha": alpha,
            "beta": beta,
            "tau_square": np.mean(tau_square_storage),
        }


#%%
from numpy.random import default_rng

rng = default_rng()
rng.exponential(2)
# %%

