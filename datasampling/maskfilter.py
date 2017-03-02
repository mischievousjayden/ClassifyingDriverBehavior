import argparse
import math
import numpy as np


def get_delta_last(size):
    """generate mask take last element
    Args:
        size (int)
    Returns:
        List<double>
    """
    mask = np.zeros(size)
    mask[size-1] = 1
    return  mask


def get_delta_mask(size):
    """generate delta mask
    Args:
        size (int)
    Returns:
        List<double>
    """
    mask = np.zeros(size)
    mask[size//2] = 1
    return  mask


def get_mean_mask(size):
    """generate delta mask
    Args:
        size (int)
    Returns:
        List<double>
    """
    mask = np.ones(size) / size
    return mask


def get_gaussian_mask(size, sigma=None):
    """generate delta mask
    Args:
        size (int)
        sigma (double)
    Returns:
        List<double>
    """
    if sigma is None:
        sigma = (size-1)/6.0

    center = size//2
    mask = np.zeros(size)
    c = (1.0/(math.sqrt(2*math.pi)*sigma))
    d = 2*sigma**2
    for i in range(size):
        mask[i] = c*math.exp((-(i-center)**2)/d)
    scale = 2.0 - sum(mask)
    mask = mask*scale
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mask", help="""select mask
            1. delta
            2. mean
            3. gaussian
            """, type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("-s", "--size", help="mask size", type=int, default=11)

    args = parser.parse_args()
    mask_type = args.mask
    size = args.size
    if(mask_type == 3):
        mask = get_gaussian_mask(size)
    elif(mask_type == 2):
        mask = get_mean_mask(size)
    else:
        mask = get_delta_mask(size)
    print(mask)


if __name__ == "__main__":
    main()

