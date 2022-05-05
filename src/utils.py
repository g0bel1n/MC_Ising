import contextlib
import operator

import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(int)


cliques = {
    4: [(1, 0), (0, 1), (-1, 0), (0, -1)],
    8: [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)],
    24: [
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (1, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (-2, -2),
        (-2, -1),
        (-2, 0),
        (-2, 1),
        (-2, 2),
        (-1, -2),
        (-1, 2),
        (0, -2),
        (0, 2),
        (1, -2),
        (1, 2),
        (2, -2),
        (2, -1),
        (2, 0),
        (2, 1),
        (2, 2),
    ],
}


def CliqueSum(
    new_value: int, ind_pixel: tuple[int, int], img: np.ndarray, connex=8
) -> float:
    """
    It takes a pixel value, an index, an image, and a connexity, and returns the number of pixels in the
    clique of the pixel at the given index that have the same value as the given pixel value

    :param new_value: the value of the pixel we're considering
    :type new_value: int
    :param ind_pixel: the index of the pixel we're currently looking at
    :type ind_pixel: tuple[int, int]
    :param img: the image to be processed
    :type img: np.ndarray
    :param connex: the number of neighbours to consider, defaults to 8 (optional)
    :return: The number of pixels in the clique that are equal to the new value.
    """
    s = 0
    for index in cliques[connex]:
        with contextlib.suppress(IndexError):
            s += int(img[tuple(map(operator.add, ind_pixel, index))] - new_value == 0)
    return s
