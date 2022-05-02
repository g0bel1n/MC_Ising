from scipy.stats import invgamma
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

    def U(self, new_value: int, ind_pixel: tuple[int, int], img: np.ndarray, alpha: float, beta: float) -> float:
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
        changes=[]
        rng = default_rng()
        L, W = img.shape
        X = img.copy()/255.
        tau_square = self.tau_square
        tau_square_storage = 0. if self.tau_square is None else self.tau_square
        alpha_storage = 0. if self.alpha is None else self.alpha
        beta_storage = 0. if self.beta is None else self.beta

        alpha = self.alpha
        beta = self.beta
        alpha_invgamma, beta_invgamma, lambda_alpha, lambda_beta = 2,1,1e7,1e7

        if gif:
            i: int = 0
            plt.imsave(f"data/output/gif/{i}.png", X, cmap="gray")
            i += 1
        with tqdm(total=self.burn_in * L * W, desc="Burn in", ascii="░▒█") as bbar:
            for _ in range(self.burn_in):
                if self.tau_square is None :
                    a,b = alpha_invgamma + (L*W)/2, beta_invgamma + np.sum(np.square(X))/2
                    tau_square = invgamma(a=a, scale = b).rvs()

                if self.alpha or self.beta is None :
                    mu_alpha, mu_beta = np.sum(X), np.sum([CliqueSum(new_value=X[l,w],ind_pixel=(l,w),img=X) for l, w in itertools.product(range(L), range(W))])
                    alpha, beta = rng.exponential(lambda_alpha- mu_alpha), rng.exponential(lambda_beta-mu_beta)
                
                change=0

                for l, w in itertools.product(range(L), range(W)):
                    probas = self.getProbas(pixel=(l, w), img=X, alpha=alpha, beta=beta, tau_square=tau_square)
                    new_x = np.random.choice((0, 1), 1, p=probas)
                    if new_x!=X[l, w]: change+=1
                    X[l, w] = new_x
                    bbar.update(1)
                    if gif and l%10==0 and w==0:
                        plt.imsave(f"data/output/gif/{i}.png", X, cmap="gray")
                        i += 1
                changes.append(change)

        avg = np.zeros_like(img).astype(np.float64)
        with tqdm(total=self.n_samples * L * W, desc="Sampling", ascii="░▒█") as pbar:
            for _ in range(self.n_samples):

                if self.tau_square is None :
                    a,b = alpha_invgamma + (L*W)/2, beta_invgamma + np.sum(np.square(X))/2
                    tau_square = invgamma(a=a, scale = b).rvs()


                if self.alpha or self.beta is None :
                    mu_alpha, mu_beta = np.sum(X), np.sum([CliqueSum(new_value=int(X[l,w]),ind_pixel=(l,w),img=X) for l, w in itertools.product(range(L), range(W))])
                    alpha, beta = rng.exponential(lambda_alpha- mu_alpha), rng.exponential(lambda_beta-mu_beta)

                tau_square_storage += float(tau_square)
                alpha_storage += float(alpha)
                beta_storage += float(beta)
                
                change=0
                for l, w in itertools.product(range(L), range(W)):
                    probas = self.getProbas(pixel=(l, w), img=X, alpha=alpha, beta=beta, tau_square=tau_square)
                    new_x = np.random.choice((0, 1), 1, p=probas)
                    if new_x!=X[l, w]: change+=1
                    X[l, w] = new_x
                    avg += X
                    pbar.update(1)
                    if gif and l%100==0 and w==0:
                        plt.imsave(f"data/output/gif/{i}.png", X, cmap="gray")
                        i += 1
                changes.append(change)

        avg = avg.astype(float)
        avg = avg / (L * W * self.n_samples)
        avg[avg >= 1.0 / 2] = 255
        avg[avg < 1.0 / 2] = 0
        avg = avg.astype(np.uint8)

        plt.plot([np.log(change) if change>0 else 0 for change in changes])
        plt.show()
        plt.savefig('changes')
        return {'img':avg, 'alpha': alpha_storage/self.n_samples), 'beta':beta_storage/(self.n_samples), 'tau_square': tau_square_storage/(self.n_samples)}