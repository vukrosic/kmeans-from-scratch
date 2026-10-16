# k-means from scratch

Code for the video **How AI Finds Groups Nobody Labeled: 96,615 Colours Into 8 (k-Means From Scratch)**.
k-means splits a pile of dots into K groups with no labels given. Choose K centres; every dot joins its closest centre; every centre moves to the average of its dots; repeat until no dot changes group. This code writes that loop with numpy alone and runs it on 150 iris flowers, 8 piles of made-up dots, 1,797 hand-written digits, two hooked half moons and the 273,280 pixels of a photo.
About 400 lines of Python 3 with the checks. The method: Stuart Lloyd, "Least squares quantization in PCM", Bell Labs 1957 (published 1982); James MacQueen, "Some methods for classification and analysis of multivariate observations", 1967. The spread-out start: David Arthur and Sergei Vassilvitskii, "k-means++: the advantages of careful seeding", 2007.

```bash
pip install numpy scikit-learn pillow
sh run_all.sh         # every number of the video, the exercise answer and the tests; expected output is in demo-output.txt
```

`run_all.sh` takes about a minute on a laptop CPU. The flowers, the digits and the photo ship with scikit-learn (`load_iris`, `load_digits`, `load_sample_image('china.jpg')`), so nothing else is downloaded. Nothing is trained: every result is distances and averages.

```python
for _ in range(300):
    groups = squared_distances(X, C).argmin(1)          # every dot joins its closest centre
    C = np.array([X[groups == j].mean(0) for j in range(k)])   # every centre moves to its dots' average
```

## Words

| In the video | Usual name | In the code |
| --- | --- | --- |
| dot, its numbers | a data point and its features | a row of `X` |
| centre | centroid | a row of `C` |
| join: each dot to its closest centre | assignment step | `assign` |
| move: each centre to its dots' average | update step | `move` |
| distance, squared | squared Euclidean distance | `squared_distances` |
| the score | inertia, within-cluster sum of squares | `score` |
| a stuck start | a local minimum | `piles.stuck_score` in `files/results.json` |
| the spread-out start | k-means++ seeding | `plus_plus_start` |
| the bend in the score line | the elbow | `elbow` in `files/results.json` |

## Files

| File | Idea | What the run shows |
| --- | --- | --- |
| `kmeans.py` | k-means with numpy only | Squared distances, join, move, score, the loop that stops when no dot changes group, a random start and the k-means++ start. |
| `run.py` | Every number in the video | Writes `files/results.json` and the photo pictures in `files/colours/`. Iris petals, starting from flowers 115, 128 and 139: the score goes 1,070.85, 102.64, 31.78, 31.64, 31.40, 31.37; dots that change group per round 55, 2, 3, 1, 0; the 3 groups agree with the 3 species on 96 % of the flowers. Best score for K = 1 … 8: 550.9, 86.39, 31.37, 19.48, 13.92, 11.04, 9.24, 7.73. 8 piles of 40 dots: the best score is 607.4, one stuck start ends at 1,246.1; of 1,000 starts, 356 random starts end within 1 % of the best, 622 k-means++ starts, 981 for the best of 10 random starts, 999 for the best of 10 k-means++ starts. Digits, 10 groups: 79 % of the pictures sit in a group where their own digit is the most common. Two half moons of 100 dots: 48 of 200 dots land in the wrong group. The photo, 640 × 427 pixels with 96,615 different colours: with 2, 4, 8, 16 and 64 colours the average pixel is off by 53.4, 33.4, 22.0, 16.2 and 9.7 (distance in red-green-blue, each 0 to 255). |
| `values.py` | Numbers for the pictures of the video | Writes `../src/values.json` for the film project. Not part of `run_all.sh`. |
| `tests.py` | Checks against scikit-learn | The same centres, groups and score as `sklearn.cluster.KMeans` (Lloyd) on the iris start and on 3 digits runs; the score never goes up from one round to the next. |
| `exercise.py` | Write the join step | See below. |
| `solution.py` | The answer | Prints `groups of [50, 48, 52] flowers, score 31.37` and `right!`. |

## What this is not

- **Not for bends.** The border between two centres is a straight line halfway between them. Groups that curve around each other, like the two half moons, get cut across.
- **Not a way to pick K.** You choose K. The elbow of the score line is a hint, not an answer.
- **Not one sure answer.** A different start can end in a worse place. Run several starts, use k-means++, keep the lowest score.

## Exercise

`exercise.py`: fill in `assign`, the join step. For every dot, work out its squared distance to each centre and pick the closest. Run `python3 exercise.py`; starting from flowers 115, 128 and 139 your score should be 31.37. The answer is in `solution.py`.

MIT license, see `LICENSE`.
