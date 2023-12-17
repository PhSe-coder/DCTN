#!/bin/bash
export PYTHONPATH=$(pwd)
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
python ./preprocess/extract.py --data-dir ./data/raw --threshold 0.3 --dest ./data
python ./preprocess/preprocess.py
python ./preprocess/split.py --src ./data/dataset --dest ./processed/tmp --seed 42 --ratio 0.8
python ./preprocess/ann.py