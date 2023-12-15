#!/bin/bash
export PYTHONPATH=$(pwd)
# python ./preprocess/extract.py --data-dir ./data/raw --threshold 0.3 --dest ./data
python ./preprocess/split.py --src ./data/dataset --dest ./processed/tmp --seed 42 --ratio 0.8
python ./preprocess/ann.py