import argparse
import os
import imageio


from src.denoiser import denoise


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

    denoise(
        alpha=args.alpha,
        beta=args.beta,
        burn_in=args.b,
        n_samples=args.ns,
        sigma=args.sigma,
        imPath=args.imp,
        gif=args.g,
    )

    if args.g:
        filenames = [el for el in os.listdir("data/output/gif") if el.endswith("png")]
        images = [
            imageio.imread(f"data/output/gif/{filename}") for filename in filenames
        ]

        imageio.mimsave("data/output/denoise.gif", images)
        for filename in filenames:
            os.remove(f"data/output/gif/{filename}")


if __name__ == "__main__":
    main()
