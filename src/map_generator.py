"""Frozen Lake problem - custom map Generator.

"""
import sys
import numpy as np
from helpers import get_abspath


def generate_map(size=10):
    """Generates a map for the Frozen Lake with size n x n and saves it to a
    text file.

    Args:
        size (int): Number of rows/columns.

    """
    lake = np.random.choice(["F", "H"], (size, size), p=[0.93, 0.07])
    lake[0, 0] = "S"
    lake[size // 2, size // 2] = "G"

    # save results to file
    file = get_abspath('frozen_lake_{}.txt'.format(size), 'data')
    with open(file, 'w') as myfile:
        for i in range(size):
            myfile.write("".join(lake[i]) + "\n")


if __name__ == '__main__':
    generate_map(size=int(sys.argv[1]))
