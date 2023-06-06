#!/bin/bash
# python config/update.py --source service --target restaurant --max_epochs 30 --dim 300
# python trainers.py fit --config config/pretrain.yaml
# python trainers.py fit --config config/train.yaml
model_path=/root/autodl-tmp/lightning_logs/*/checkpoints/fdgr-*.ckpt
declare -a paths=($(ls -v $model_path)) # list all the model checkpoints
python trainers.py test --config config/test.yaml --ckpt_path ${paths[-1]}