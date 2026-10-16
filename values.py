"""Compact data for the film's pictures: reads files/results.json, adds the few
things run.py does not keep (which of the 1,000 tries found the best score,
the digit pictures shown, a sample of the photo's colours), writes
../src/values.json and copies the colour pictures to ../public/img."""
import json
import os
import shutil

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits, load_sample_image

from kmeans import kmeans, random_start, plus_plus_start

R = json.load(open('files/results.json'))
rng = np.random.default_rng(3)                          # run.py's piles, unrounded
middles = np.array([[x, y] for y in (0, 6) for x in (0, 6, 12, 18)], float)
P = np.concatenate([m + rng.normal(0, 1.0, (40, 2)) for m in middles])
assert np.allclose(P.round(3), R['piles']['points'])
best = R['piles']['best']

# the 1,000 tries of run.py again (same seed, same order): which ones found the best score
good = lambda s: s < best * 1.01
r = np.random.default_rng(7)
rand = [good(kmeans(P, random_start(P, 8, r))[2][-1]['score']) for _ in range(1000)]
pp = [good(kmeans(P, plus_plus_start(P, 8, r))[2][-1]['score']) for _ in range(1000)]
ten = [good(min(kmeans(P, random_start(P, 8, r))[2][-1]['score'] for _ in range(10))) for _ in range(1000)]
ten_pp = [good(min(kmeans(P, plus_plus_start(P, 8, r))[2][-1]['score'] for _ in range(10))) for _ in range(1000)]
assert (sum(rand), sum(pp), sum(ten), sum(ten_pp)) == (R['piles']['random_good'], R['piles']['pp_good'],
                                                        R['piles']['ten_good'], R['piles']['ten_pp_good'])
bits = lambda v: ''.join('1' if g else '0' for g in v)

# the digit pictures shown: one example, and a second one for the square-by-square distance
X, y = load_digits(return_X_y=True)

# moons: which dots land in the wrong group
m = R['moons']
g = np.array(m['groups'])
truth = np.repeat([0, 1], 100)
flip = (g == truth).mean() < 0.5
wrong = np.where((g == truth) if flip else (g != truth))[0]
assert len(wrong) == m['wrong']

# colours: a sample of the photo's pixels, drawn flat with the two main directions of the colour cloud
img = load_sample_image('china.jpg')
px = img.reshape(-1, 3).astype(float)
rng = np.random.default_rng(1)
pick = px[rng.choice(len(px), 900, replace=False)]
mu = px.mean(0)
_, _, Vt = np.linalg.svd(pick - mu, full_matrices=False)
flat = lambda Z: ((np.asarray(Z, float) - mu) @ Vt[:2].T).round(1).tolist()
roof = [int(v) for v in img[262, 250]]                 # one pixel on the orange roof, row 262, column 250
crop = img[259:265, 245:255].astype(int).tolist()     # 6 rows × 10 columns of pixels around it

c = R['colours']
os.makedirs('../public/img', exist_ok=True)
for name in ('original', 'k2', 'k4', 'k8', 'k16', 'k64'):
    shutil.copy(f'files/colours/{name}.png', f'../public/img/{name}.png')

f, l, p = R['flowers'], R['loop'], R['piles']
out = {
    'flowers': {'points': f['points'], 'species': f['species'], 'one': f['one'], 'pair': f['pair'], 'starts': f['starts']},
    'loop': {'start': l['start'], 'rounds': [{'c': q['centres'], 'g': q['groups'], 's': q['score'], 'n': q['sizes']} for q in l['rounds']],
             'dot': l['dot'], 'moved': l['moved'], 'changed': l['changed'], 'table': l['table'], 'agree': l['agree']},
    'avg': R['average_demo'],
    'elbow': R['elbow'],
    'piles': {k: p[k] for k in ('points', 'best', 'stuck_start', 'stuck_centres', 'stuck_groups', 'stuck_score',
                                'stuck_rounds', 'random_good', 'pp_good', 'ten_good', 'ten_pp_good', 'pp_demo',
                                'pp_start', 'pp_final')},
    'tries': {'random': bits(rand), 'pp': bits(pp), 'ten': bits(ten), 'ten_pp': bits(ten_pp)},
    'digits': {'n': R['digits']['n'], 'agree': R['digits']['agree'], 'rounds': R['digits']['rounds'],
               'groups': R['digits']['groups'], 'ones': R['digits']['ones'],
               'pic': X[0].astype(int).tolist(), 'pic_digit': int(y[0]),
               'pic2': X[1].astype(int).tolist(), 'pic2_digit': int(y[1])},
    'moons': {'points': m['points'], 'groups': m['groups'], 'centres': m['centres'], 'wrong': wrong.tolist(), 'agree': m['agree']},
    'colours': {'width': c['width'], 'height': c['height'], 'pixels': c['pixels'], 'different': c['different'],
                'k': {k: {'palette': c[k]['palette'], 'off': c[k]['off'], 'bits': c[k]['bits']} for k in ('2', '4', '8', '16', '64')},
                'sample': pick.astype(int).tolist(), 'flat': flat(pick),
                'flat8': flat(c['8']['palette']), 'roof': roof, 'crop': crop},
}
json.dump(out, open('../src/values.json', 'w'), separators=(',', ':'))
print('values.json', os.path.getsize('../src/values.json') // 1024, 'kB; roof pixel', roof,
      'digit', out['digits']['pic_digit'], out['digits']['pic2_digit'], 'wrong', len(wrong))
