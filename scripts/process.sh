#!/bin/bash
export PYTHONPATH=$(pwd)
python ./preprocess/split.py --seed 42 --ratio 0.7
python ./preprocess/ann.py