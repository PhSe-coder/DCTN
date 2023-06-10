#!/bin/bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# python trainers.py fit --config config/pretrain-source.yaml
for p in 1.0 0.8 0.6 0.5 0.4 0.2 0
do
    python config/update.py --source service --target restaurant --max_pretrain_epochs 15 --max_train_epochs 15 --dim 300 --p $p
    python trainers.py fit --config config/pretrain-target.yaml --trainer.enable_progress_bar false
    for coff in 0.01 0.02 0.03 0.04 0.05 0.08 1
    do
        python trainers.py fit --config config/train.yaml --model.init_args.coff $coff --trainer.enable_progress_bar false
        model_path=/root/autodl-tmp/lightning_logs/*/checkpoints/fdgr-*.ckpt
        declare -a paths=($(ls -v $model_path)) # list all the model checkpoints
        python trainers.py test --config config/test.yaml --ckpt_path ${paths[-1]} --trainer.enable_progress_bar false
        echo "p: $p, coff: $coff"
    done
done