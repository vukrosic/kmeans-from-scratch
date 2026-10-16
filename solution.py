"""The answer to exercise.py.   python3 solution.py"""
import numpy as np
from sklearn.datasets import load_iris


def assign(X, C):
    """X: the dots, one row each (150 x 2). C: the centres (3 x 2).
    Return, for every dot, the number of its closest centre (0, 1 or 2)."""
    groups = np.zeros(len(X), dtype=int)
    for i, dot in enumerate(X):
        squared = ((C - dot) ** 2).sum(1)    # 1. across squared + up squared, per centre
        groups[i] = squared.argmin()         # 2. the closest centre
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
