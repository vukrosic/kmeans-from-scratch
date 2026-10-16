"""Every number in the film. Writes files/results.json and the colour pictures
in files/colours/. About half a minute on a laptop."""
import json
import os
import time

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits, load_iris, load_sample_image

from kmeans import squared_distances, assign, move, score, kmeans, random_start, plus_plus_start


def best_of(X, k, starts, rng):
    """Run k-means from `starts` k-means++ starts and keep the lowest score."""
    runs = [kmeans(X, plus_plus_start(X, k, rng)) for _ in range(starts)]
    return min(runs, key=lambda r: r[2][-1]['score'])


def flowers(out):
    """150 iris flowers, two numbers each: petal length and petal width (cm)."""
    iris = load_iris()
    X, species = iris.data[:, 2:], iris.target
    out['flowers'] = {'n': len(X), 'points': X.tolist(), 'species': species.tolist(),
                      'species_names': [str(s) for s in iris.target_names], 'one': X[0].tolist()}

    # distance between two flowers, by hand
    a, b = X[0], X[100]
    out['flowers']['pair'] = {'a': a.tolist(), 'b': b.tolist(), 'across': float(b[0] - a[0]),
                              'up': float(b[1] - a[1]), 'squared': float(((b - a) ** 2).sum()),
                              'distance': float(np.linalg.norm(b - a))}

    # the loop from three flowers in the same corner
    start = [115, 128, 139]
    C, groups, hist = kmeans(X, X[start])
    rounds = []
    g = None
    for h in hist:
        g = assign(X, h['centres'])
        rounds.append({'centres': h['centres'].round(3).tolist(), 'groups': g.tolist(),
                       'score': h['score'], 'sizes': np.bincount(g, minlength=3).tolist()})
    out['loop'] = {'start': start, 'rounds': rounds}

    # one dot's three distances in round one, and one centre's move
    dot = 60
    d2 = squared_distances(X[dot:dot + 1], X[start])[0]
    g1 = assign(X, X[start])
    j = int(np.bincount(g1).argmax())
    members = X[g1 == j]
    out['loop']['dot'] = {'id': dot, 'xy': X[dot].tolist(), 'squared': d2.tolist(),
                          'distance': np.sqrt(d2).tolist(), 'joins': int(d2.argmin())}
    out['loop']['moved'] = {'centre': j, 'from': X[start][j].tolist(), 'members': int(len(members)),
                            'sum': members.sum(0).tolist(), 'to': members.mean(0).tolist()}

    # how many dots changed group each round
    changed = [int((np.array(rounds[i]['groups']) != np.array(rounds[i - 1]['groups'])).sum())
               for i in range(1, len(rounds))]
    out['loop']['changed'] = changed

    # check against the hidden species
    table = [[int(((groups == gi) & (species == s)).sum()) for s in range(3)] for gi in range(3)]
    out['loop']['table'] = table
    out['loop']['agree'] = sum(max(r) for r in table) / len(X)

    # why the average: three numbers on a line
    v = np.array([1.0, 2.0, 6.0])
    out['average_demo'] = {'values': v.tolist(), 'average': float(v.mean()),
                           'at': {str(p): float(((v - p) ** 2).sum()) for p in (2, 3, 4)}}

    # does the start matter on the flowers? 1,000 random starts
    r = np.random.default_rng(8)
    s = np.array([kmeans(X, random_start(X, 3, r))[2][-1]['score'] for _ in range(1000)])
    out['flowers']['starts'] = {'n': 1000, 'lowest': float(s.min()), 'highest': float(s.max()),
                                'at_lowest': int((s < s.min() + 0.01).sum()),
                                'within_tenth': int((s < s.min() + 0.1).sum())}

    # how many groups: best of 10 starts for 1 to 8 groups
    rng = np.random.default_rng(5)
    out['elbow'] = {k: best_of(X, k, 10, rng)[2][-1]['score'] for k in range(1, 9)}
    out['elbow']['150'] = 0.0
    return X


def piles(out):
    """320 dots in 8 piles made up for this test, 40 dots in each."""
    rng = np.random.default_rng(3)
    middles = np.array([[x, y] for y in (0, 6) for x in (0, 6, 12, 18)], float)
    P = np.concatenate([m + rng.normal(0, 1.0, (40, 2)) for m in middles])
    best = best_of(P, 8, 20, np.random.default_rng(0))[2][-1]['score']

    # one random start that gets stuck, shown in the film
    for seed in range(100):
        r = np.random.default_rng(seed)
        S = random_start(P, 8, r)
        C, g, h = kmeans(P, S)
        per_pile = [np.bincount(g[i * 40:(i + 1) * 40], minlength=8) for i in range(8)]
        shared = [i for i in range(8) if (per_pile[i] > 0).sum() == 1 and
                  sum(per_pile[m][per_pile[i].argmax()] > 20 for m in range(8)) == 2]
        if h[-1]['score'] > best * 1.3 and len(h) >= 4 and shared:
            break
    out['piles'] = {'points': P.round(3).tolist(), 'best': best, 'stuck_seed': seed,
                    'stuck_start': S.round(3).tolist(), 'stuck_centres': C.round(3).tolist(),
                    'stuck_groups': g.tolist(), 'stuck_score': h[-1]['score'], 'stuck_rounds': len(h)}

    # 1,000 random starts, 1,000 k-means++ starts, 1,000 times best of 10 random starts
    good = lambda s: s < best * 1.01
    r = np.random.default_rng(7)
    rand = [kmeans(P, random_start(P, 8, r))[2][-1]['score'] for _ in range(1000)]
    pp = [kmeans(P, plus_plus_start(P, 8, r))[2][-1]['score'] for _ in range(1000)]
    ten = [min(kmeans(P, random_start(P, 8, r))[2][-1]['score'] for _ in range(10)) for _ in range(1000)]
    ten_pp = [min(kmeans(P, plus_plus_start(P, 8, r))[2][-1]['score'] for _ in range(10)) for _ in range(1000)]
    out['piles'].update({'random_good': int(sum(map(good, rand))), 'pp_good': int(sum(map(good, pp))),
                         'ten_good': int(sum(map(good, ten))), 'ten_pp_good': int(sum(map(good, ten_pp))),
                         'random_mean': float(np.mean(rand)),
                         'pp_mean': float(np.mean(pp))})

    # k-means++ by hand: the first centre, then each dot's chance of being the second
    pile = lambda c: int(np.argmin(((middles - c) ** 2).sum(1)))
    for seed in range(11, 200):                  # a start that lands in all 8 piles, for the picture
        r = np.random.default_rng(seed)
        if len({pile(c) for c in plus_plus_start(P, 8, r)}) == 8:
            break
    r = np.random.default_rng(seed)
    first = int(r.integers(len(P)))
    d = squared_distances(P, P[first:first + 1])[:, 0]
    chance = d / d.sum()
    near = int(np.argsort(d)[5])
    far = int(d.argmax())
    out['piles']['pp_demo'] = {'seed': seed, 'first': first, 'near': near, 'far': far,
                               'near_sq': float(d[near]), 'far_sq': float(d[far]), 'sum_sq': float(d.sum()),
                               'near_chance': float(chance[near]), 'far_chance': float(chance[far]),
                               'own_pile_chance': float(chance[(first // 40) * 40:(first // 40 + 1) * 40].sum())}
    C = [P[first]]
    for _ in range(7):
        dd = squared_distances(P, np.array(C)).min(1)
        C.append(P[r.choice(len(P), p=dd / dd.sum())])
    C = np.array(C)
    out['piles']['pp_start'] = C.round(3).tolist()
    out['piles']['pp_start_piles'] = sorted(pile(c) for c in C)
    Cf, gf, hf = kmeans(P, C)
    out['piles']['pp_final'] = hf[-1]['score']


def digits(out):
    """1,797 pictures of hand-written digits, 8 by 8 squares, 64 numbers each."""
    X, y = load_digits(return_X_y=True)
    C, g, h = best_of(X, 10, 10, np.random.default_rng(2))
    groups = []
    ones = {}                                    # the ones split over two groups: their average picture in each
    for j in range(10):
        if (y[g == j] == 1).sum() > 40:
            most = int(np.bincount(y[g == j]).argmax())
            ones[str(most)] = {'count': int((y[g == j] == 1).sum()),
                               'ones_average': X[(g == j) & (y == 1)].mean(0).round(2).tolist(),
                               'eights_average': X[(g == j) & (y == 8)].mean(0).round(2).tolist() if most == 8 else None}
        counts = np.bincount(y[g == j], minlength=10)
        groups.append({'centre': C[j].round(2).tolist(), 'size': int(counts.sum()),
                       'most': int(counts.argmax()), 'most_count': int(counts.max()),
                       'second': int(np.argsort(counts)[-2]), 'second_count': int(np.sort(counts)[-2])})
    groups.sort(key=lambda r: (r['most'], -r['most_count']))
    out['digits'] = {'n': len(X), 'groups': groups, 'score': h[-1]['score'], 'rounds': len(h),
                     'agree': sum(r['most_count'] for r in groups) / len(X), 'ones': ones,
                     'no_group_of_own': sorted(set(range(10)) - {r['most'] for r in groups})}


def moons(out):
    """Two half circles hooked into each other, 100 dots each."""
    rng = np.random.default_rng(3)
    t = np.linspace(0, np.pi, 100)
    M = np.r_[np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)]]
    M = M + rng.normal(0, 0.06, M.shape)
    truth = np.repeat([0, 1], 100)
    C, g, h = best_of(M, 2, 10, np.random.default_rng(4))
    agree = max((g == truth).mean(), (g != truth).mean())
    out['moons'] = {'points': M.round(3).tolist(), 'groups': g.tolist(), 'centres': C.round(3).tolist(),
                    'agree': float(agree), 'wrong': int(round((1 - agree) * 200))}


def colours(out):
    """A photo's pixels as dots with 3 numbers each (red, green, blue, 0 to 255).
    k-means on 5,000 random pixels; then every pixel takes its centre's colour."""
    img = load_sample_image('china.jpg')
    H, W, _ = img.shape
    px = img.reshape(-1, 3).astype(float)
    os.makedirs('files/colours', exist_ok=True)
    Image.fromarray(img).save('files/colours/original.png')
    res = {'height': H, 'width': W, 'pixels': H * W, 'different': int(len(np.unique(px, axis=0)))}
    rng = np.random.default_rng(6)
    sample = px[rng.choice(len(px), 5000, replace=False)]
    for k in (2, 4, 8, 16, 64):
        C, _, h = kmeans(sample, plus_plus_start(sample, k, rng))
        g = assign(px, C)
        new = C[g].round().clip(0, 255).astype(np.uint8)
        Image.fromarray(new.reshape(H, W, 3)).save(f'files/colours/k{k}.png')
        res[str(k)] = {'palette': C.round().astype(int).tolist(), 'rounds': len(h),
                       'off': float(np.sqrt(((new - px) ** 2).sum(1)).mean()),
                       'bits': int(np.log2(k))}
    out['colours'] = res


def main():
    t0 = time.time()
    out = {}
    flowers(out)
    piles(out)
    digits(out)
    moons(out)
    colours(out)
    out['seconds'] = time.time() - t0
    json.dump(out, open('files/results.json', 'w'))
    f, l, p = out['flowers'], out['loop'], out['piles']
    print('flower one', f['one'], 'pair', {k: v for k, v in f['pair'].items()})
    print('loop scores', [round(r['score'], 2) for r in l['rounds']], 'sizes', [r['sizes'] for r in l['rounds']])
    print('changed', l['changed'], 'dot', l['dot'], 'moved', l['moved'])
    print('table', l['table'], 'agree', l['agree'], 'average', out['average_demo'])
    print('elbow', {k: round(v, 2) for k, v in out['elbow'].items()})
    print('piles', {k: v for k, v in p.items() if k not in ('points', 'stuck_start', 'stuck_centres', 'stuck_groups', 'pp_start')})
    d = out['digits']
    print('digits', d['score'], d['rounds'], d['agree'], d['no_group_of_own'],
          [(r['most'], r['most_count'], r['size'], r['second'], r['second_count']) for r in d['groups']])
    print('moons', out['moons']['agree'], out['moons']['wrong'])
    c = out['colours']
    print('colours', {k: (v if not isinstance(v, dict) else (v['rounds'], round(v['off'], 1), v['bits'])) for k, v in c.items()})
    print('seconds', round(out['seconds'], 1))


if __name__ == '__main__':
    main()
