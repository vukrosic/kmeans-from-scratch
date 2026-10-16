"""Exercise: write the assign step yourself.

Fill in assign below, then run:   python3 exercise.py
It runs k-means on the 150 flowers with your assign step and checks the
answer. The answer is in solution.py."""
import numpy as np
from sklearn.datasets import load_iris


def assign(X, C):
    """X: the dots, one row each (150 x 2). C: the centres (3 x 2).
    Return, for every dot, the number of its closest centre (0, 1 or 2)."""
    groups = np.zeros(len(X), dtype=int)
    # your part starts here
    # for every dot:
    #   1. its squared distance to each centre: across squared + up squared
    #   2. the centre with the smallest one is its group
    # and ends here
    return groups


def run(X, C):
    groups = None
    while True:
        new = assign(X, C)
        if groups is not None and (new == groups).all():
            return C, new
        groups = new
        C = np.array([X[groups == j].mean(0) if (groups == j).any() else C[j] for j in range(len(C))])


if __name__ == '__main__':
    X = load_iris().data[:, 2:]
    C, groups = run(X, X[[115, 128, 139]])
    s = ((X - C[groups]) ** 2).sum()
    print(f'groups of {np.bincount(groups, minlength=3).tolist()} flowers, score {s:.2f}')
    print('right!' if abs(s - 31.37) < 0.01 else 'not yet: the score should be 31.37')
