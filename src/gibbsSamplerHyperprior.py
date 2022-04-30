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
        changes=[]
        L, W = img.shape
        X = img.copy()

        if gif:
            i: int = 0
            plt.imsave(f"data/output/gif/{i}.png", X, cmap="gray")
            i += 1
        with tqdm(total=self.burn_in * L * W, desc="Burn in", ascii="░▒█") as bbar:
            for _ in range(self.burn_in):
                change=0
                for l, w in itertools.product(range(L), range(W)):
                    probas = self.getProbas(pixel=(l, w), img=X)
                    new_x = np.random.choice((0, 255), 1, p=probas)
                    if new_x!=X[l, w]: change+=1
                    X[l, w] = new_x
                    bbar.update(1)
                    if gif and l%10==0 and w==0:
                        plt.imsave(f"data/output/gif/{i}.png", X, cmap="gray")
                        i += 1
                changes.append(change)

        avg = np.zeros_like(img).astype(np.uint64)
        with tqdm(total=self.n_samples * L * W, desc="Sampling", ascii="░▒█") as pbar:
            for _ in range(self.n_samples):
                change=0
                for l, w in itertools.product(range(L), range(W)):
                    probas = self.getProbas(pixel=(l, w), img=X)
                    new_x = np.random.choice((0, 255), 1, p=probas)
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
        avg[avg >= 255.0 / 2] = 255
        avg[avg < 255.0 / 2] = 0
        avg = avg.astype(np.uint8)

        plt.plot([np.log(change) if change>0 else 0 for change in changes])
        plt.show()
        plt.savefig('changes')
        return avg