#!/bin/bash
export TRANSFORMERS_OFFLINE=1
domains=('restaurant' 'laptop' 'twitter')
seeds=(42 1000 100 2024 1119)
set -e
for seed in ${seeds[@]};
do
for tar_domain in ${domains[@]};
do
    for src_domain in ${domains[@]};
    do
        if [ $src_domain != $tar_domain ];
        then
            echo "Train: $src_domain -> $tar_domain seed: $seed"
            python trainers.py \
            --config config/train.yaml \
            --data.init_args.train_file ./processed/dataset/$src_domain.train.txt \
            --data.init_args.contrast_file ./processed/dataset/$src_domain.contrast.train.txt \
            --data.init_args.validation_file ./processed/dataset/$src_domain.validation.txt \
            --data.init_args.test_file ./processed/dataset/$tar_domain.test.txt \
            --seed_everything $seed
        fi
        rm -rf /root/autodl-tmp/lightning_logs/*
    done
done

done
