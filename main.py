import argparse
import os

import imageio
import matplotlib.pyplot as plt

from src.gibbSamplerVariance import GibbsSampler
from src.noiser import Noiser


def main():
    """
    It takes an image, adds noise to it, and then denoises it
    """

    # Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=40, help="Burn in")
    parser.add_argument("--ns", type=int, default=5, help="Samples")
    parser.add_argument(
        "--imp",
        type=str,
        default="data/input/test_img.jpeg",
        help="Path of the image to noise",
    )
    parser.add_argument("--alpha", type=float, default=0.0, help="alpha")
    parser.add_argument("--beta", type=float, default=1.3, help="beta")
    parser.add_argument("--sigma", type=float, default=179, help="sigma")
    parser.add_argument("--findsigma", type=bool, default=False, help="find sigma ?")

    parser.add_argument("--g", type=bool, default=True, help="Save for gif")

    args = parser.parse_args()

    gs = GibbsSampler(
        args.alpha,
        args.beta,
        None if args.findsigma else args.sigma,
        burn_in=args.b,
        n_samples=args.ns,
    )

    true_image = plt.imread(args.imp)  # Raw image

    noised_img = Noiser(args.sigma).noise_img(imPath=args.imp)  # Noised Image

    results = gs.sample(noised_img, gif=args.g)  # Dict of results

    denoised = results["img"]  # Denoised Image

    print(f"{results['alpha']=} \n{results['beta']=} \n{results['tau_square']=} \n")

    # Plotting...
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(true_image, cmap="gray")
    ax1.set_title("Image initiale")
    ax2.imshow(noised_img, cmap="gray")
    ax2.set_title("Image bruitée")
    ax3.imshow(denoised, cmap="gray")
    ax3.set_title("Image débruitée")

    plt.savefig(
        "data/output/ising.jpg"
    )  # Saves a plot containing the raw, the noised and the denoised image.

    # To make gifs
    if args.g:

        filenames = [
            int(el[:-4]) for el in os.listdir("data/output/gif") if el.endswith("png")
        ]
        filenames.sort()
        images = [
            imageio.imread(f"data/output/gif/{filename}.png") for filename in filenames
        ]

        imageio.mimsave("data/output/denoise.gif", images, fps=25)
        for filename in filenames:
            os.remove(f"data/output/gif/{filename}.png")


if __name__ == "__main__":
    main()
