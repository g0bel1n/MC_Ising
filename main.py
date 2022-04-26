import argparse

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
    parser.add_argument("--alpha", type=float, default=0.01, help="alpha")
    parser.add_argument("--beta", type=float, default=10, help="beta")
    parser.add_argument("--sigma", type=float, default=120, help="sigma")

    args = parser.parse_args()

    denoise(
        alpha=args.alpha,
        beta=args.beta,
        burn_in=args.b,
        n_samples=args.ns,
        sigma=args.sigma,
        imPath=args.imp,
    )


if __name__ == "__main__":
    main()
