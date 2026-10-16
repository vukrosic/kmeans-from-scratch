"""Checks that the from-scratch k-means agrees with scikit-learn.   python3 tests.py"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, load_digits

from kmeans import kmeans, plus_plus_start, score, assign


def main():
    X = load_iris().data[:, 2:]
    start = X[[115, 128, 139]]
    C, g, hist = kmeans(X, start)
    ref = KMeans(3, init=start, n_init=1, tol=0, algorithm='lloyd').fit(X)
    assert np.allclose(C, ref.cluster_centers_), (C, ref.cluster_centers_)
    assert (g == ref.labels_).all()
    assert abs(score(X, g, C) - ref.inertia_) < 1e-9, (score(X, g, C), ref.inertia_)
    s = [h['score'] for h in hist]
    assert all(b <= a + 1e-9 for a, b in zip(s, s[1:])), s       # the score never goes up

    D, _ = load_digits(return_X_y=True)
    rng = np.random.default_rng(0)
    for _ in range(3):
        S = plus_plus_start(D, 10, rng)
        C, g, hist = kmeans(D, S)
        ref = KMeans(10, init=S, n_init=1, tol=0, algorithm='lloyd', max_iter=300).fit(D)
        assert abs(score(D, g, C) - ref.inertia_) / ref.inertia_ < 1e-9
        assert (assign(D, C) == g).all()                           # stopped: nothing would change
    print('all tests pass')


if __name__ == '__main__':
    main()
