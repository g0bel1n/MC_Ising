from matplotlib import pyplot as plt

from src.noiser import Noiser
from src.sampler import GibbsSampler


def denoise(
    alpha: float, beta: float, burn_in: int, n_samples: int, sigma: float, imPath: str
):

    gs = GibbsSampler(alpha, beta, sigma, burn_in=burn_in, n_samples=n_samples)

    # Loading of images
    noised_img = Noiser(sigma).noise_img(imPath=imPath)[:224, :224]
    true_image = plt.imread(imPath)[:224, :224]
    denoised = gs.sample(noised_img)

    # Plotting...
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(true_image, cmap="gray")
    ax1.set_title("Image initiale")
    ax2.imshow(noised_img, cmap="gray")
    ax2.set_title("Image bruitée")
    ax3.imshow(denoised, cmap="gray")
    ax3.set_title("Image débruitée")

    plt.savefig("data/output/ising.jpg")
