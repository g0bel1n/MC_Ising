from scipy.stats import invgamma
from statsmodels.graphics.tsaplots import plot_acf
from numpy.random import default_rng
from src.sampler import Sampler
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

        ###########################################
        # Initialisation of the global parameters #
        ###########################################

        changes = []
        L, W = img.shape
        rng = default_rng()
        indices = list(itertools.product(range(L), range(W)))
        change_rate = 0.6
        X = img.copy() / 255.0
        tau_square = self.tau_square
        tau_square_storage = [0.0] if self.tau_square is None else [self.tau_square]
        x_storage = [X[0, 0]]

        alpha_invgamma, beta_invgamma = 2, 100

        if gif:
            i: int = 0
            plt.imsave(f"data/output/gif/{i}.png", X, cmap="gray")
            i += 1

        ###########################################
        #           Start of Burn in loop         #
        ###########################################

        with tqdm(
            total=self.burn_in * L * W * change_rate, desc="Burn in", ascii="░▒█"
        ) as bbar:
            for _ in range(self.burn_in):
                if self.tau_square is None:  # Sampling tau if it is not given
                    a, b = (
                        alpha_invgamma + (L * W) / 2,  # Updating the hyperparameters
                        beta_invgamma
                        + np.sum(np.square(img / 255.0 - X))
                        / 2,  # Updating the hyperparameters
                    )
                    tau_square = invgamma(a=a, scale=b).rvs()

                change = 0  # Number of change in a iteration

                x_storage.append(X[0, 0])  # For ACF

                for l, w in rng.choice(
                    indices, int(L * W * change_rate)
                ):  # randomly selecting  change_rate% amongst all pixels
                    probas = self.getProbas(
                        pixel=(l, w),
                        img=X,
                        alpha=self.alpha,
                        beta=self.beta,
                        tau_square=tau_square,
                    )
                    new_x = np.random.choice(
                        (0, 1), 1, p=probas
                    )  # Bernoulli # new_x serves to check if the pixel changed

                    if new_x != X[l, w]:
                        change += 1

                    X[l, w] = new_x

                    bbar.update(1)

                    if gif and l % 10 == 0 and w == 0:  # Saving pic for gif
                        plt.imsave(f"data/output/gif/{i}.png", X, cmap="gray")
                        i += 1
                changes.append(change)

            ###########################################
            #           END of Burn in loop           #
            ###########################################

        avg = np.zeros_like(img).astype(np.float64)

        ###########################################
        #           Start of Main loop            #
        ###########################################

        with tqdm(
            total=self.n_samples * L * W * change_rate, desc="Sampling", ascii="░▒█"
        ) as pbar:

            for _ in range(self.n_samples):

                if self.tau_square is None:  # Sampling tau if it is not given
                    a, b = (
                        alpha_invgamma + (L * W) / 2,  # Updating the hyperparameters
                        beta_invgamma
                        + np.sum(np.square(img / 255.0 - X))
                        / 2,  # Updating the hyperparameters
                    )
                    tau_square = invgamma(a=a, scale=b).rvs()

                tau_square_storage.append(tau_square)  # Storing sample for estimation

                change = 0  # Number of change in an iteration
                x_storage.append(X[0, 0])  # For ACF
                for l, w in rng.choice(
                    indices, int(L * W * change_rate)
                ):  # randomly selecting  change_rate% amongst all pixels

                    probas = self.getProbas(
                        pixel=(l, w),
                        img=X,
                        alpha=self.alpha,
                        beta=self.beta,
                        tau_square=tau_square,
                    )

                    new_x = np.random.choice(
                        (0, 1), 1, p=probas
                    )  # Bernoulli # new_x serves to check if the pixel changed
                    if new_x != X[l, w]:
                        change += 1

                    X[l, w] = new_x

                    avg += X

                    pbar.update(1)

                    if gif and l % 20 == 0 and w == 0:
                        plt.imsave(f"data/output/gif/{i}.png", X, cmap="gray")
                        i += 1
                changes.append(change)

        ###########################################
        #           End of Main loop              #
        ###########################################

        # COmputation of the resulting image

        avg = avg.astype(float)
        avg = avg / (L * W * int(self.n_samples * change_rate))
        avg[avg >= 1.0 / 2] = 255
        avg[avg < 1.0 / 2] = 0
        avg = avg.astype(np.uint8)

        # Saving the log(changes) curve
        plt.plot([np.log(change) if change > 0 else 0 for change in changes])
        plt.show()
        plt.savefig("data/output/changes")

        # Saving the acf
        plot_acf(np.array(x_storage))
        plt.savefig("data/output/acf")

        return {
            "img": avg,
            "alpha": self.alpha,
            "beta": self.beta,
            "tau_square": np.mean(tau_square_storage),
        }
