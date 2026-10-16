"""k-means from scratch with numpy only.

Each dot is a row of numbers. A centre is a point in the same space.
One round: every dot joins its closest centre (assign), then every centre
moves to the average of its dots (move). The score is each dot's distance
to its centre, squared, all added up. Every round the score goes down or
stays; when no dot changes group, it stops."""
import warnings

import numpy as np

warnings.filterwarnings("ignore", "divide by zero|overflow|invalid value", RuntimeWarning)  # spurious on Apple matmul


def squared_distances(X, C):
    """Table of squared distances: row = dot, column = centre."""
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)


def assign(X, C):
    """Every dot joins the closest centre. Returns the group of each dot."""
    return squared_distances(X, C).argmin(1)


def move(X, groups, C):
    """Every centre moves to the average of its dots (an empty group stays)."""
    return np.array([X[groups == j].mean(0) if (groups == j).any() else C[j]
                     for j in range(len(C))])


def score(X, groups, C):
    """Each dot's distance to its own centre, squared, all added up."""
    return float(((X - C[groups]) ** 2).sum())


def kmeans(X, C, rounds=300):
    """Lloyd's loop from the starting centres C. Returns the final centres,
    groups and a history with the score of every round."""
    C = np.asarray(C, float)
    history = []
    groups = None
    for _ in range(rounds):
        new = assign(X, C)
        history.append({'centres': C.copy(), 'score': score(X, new, C)})
        if groups is not None and (new == groups).all():
            break                                   # no dot changed group
        groups = new
        C = move(X, groups, C)
    return C, groups, history


def random_start(X, k, rng):
    """k different dots, picked at random, as the first centres."""
    return X[rng.choice(len(X), k, replace=False)]


def plus_plus_start(X, k, rng):
    """k-means++: the first centre is a random dot. Each next centre is a
    dot picked with chance in proportion to its squared distance to the
    closest centre chosen so far, so far-away dots are likely."""
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = squared_distances(X, np.array(C)).min(1)
        C.append(X[rng.choice(len(X), p=d / d.sum())])
    return np.array(C)
