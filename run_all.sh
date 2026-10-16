#!/bin/bash
# Every result of the film, then the exercise answer and the tests. About a minute.
set -e
cd "$(dirname "$0")"
python3 run.py | tee files/run.out
python3 solution.py
python3 tests.py
