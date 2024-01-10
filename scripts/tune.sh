#!/bin/bash
export TRANSFORMERS_OFFLINE=1
for h_dim in 100 200 300
do
    for epoch in 5 10 15 20 25
    do
        echo "h_dim: $h_dim, epoch: $epoch"
        python trainers.py --config config/train.yaml \
                           --model.init_args.h_dim $h_dim \
                           --trainer.enable_progress_bar false \
                           --trainer.max_epochs $epoch
    done
done