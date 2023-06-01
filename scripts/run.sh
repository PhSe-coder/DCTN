#!/bin/bash
python trainers.py fit --config config/pretrain.yaml
python trainers.py fit --config config/train.yaml
python trainers.py test --config config/test.yaml