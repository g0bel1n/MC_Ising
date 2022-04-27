import argparse
import os

import imageio
import matplotlib.pyplot as plt

from src.noiser import Noiser
from src.sampler import GibbsSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=20, help="Burn in")
    parser.add_argument("--ns", type=int, default=20, help="Samples")
    parser.add_argument(
        "--imp",
        type=str,
        default="data/input/test_img.jpeg",
        help="Path of the image to noise",
    )
    parser.add_argument("--alpha", type=float, default=0.0, help="alpha")
    parser.add_argument("--beta", type=float, default=0.03, help="beta")
    parser.add_argument("--sigma", type=float, default=179, help="sigma")
    parser.add_argument("--g", type=bool, default=True, help="Save for gif")

    args = parser.parse_args()

    gs = GibbsSampler(args.alpha, args.beta, args.sigma, burn_in=args.b, n_samples=args.ns)

    # Loading of images
    noised_img = Noiser(args.sigma).noise_img(imPath=args.imp)
    true_image = plt.imread(args.imp)
    denoised = gs.sample(noised_img, gif=args.g)

    # Plotting...
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(true_image, cmap="gray")
    ax1.set_title("Image initiale")
    ax2.imshow(noised_img, cmap="gray")
    ax2.set_title("Image bruitée")
    ax3.imshow(denoised, cmap="gray")
    ax3.set_title("Image débruitée")

    plt.savefig("data/output/ising.jpg")

    if args.g:
        filenames = [int(el[:-4]) for el in os.listdir("data/output/gif") if el.endswith("png")]
        filenames.sort()
        images = [
            imageio.imread(f"data/output/gif/{filename}.png") for filename in filenames
        ]

        imageio.mimsave("data/output/denoise.gif", images, fps=3)
        for filename in filenames:
            os.remove(f"data/output/gif/{filename}.png")


if __name__ == "__main__":
    main()
