#!/bin/bash
domains=('restaurant' 'laptop' 'twitter')
export TRANSFORMERS_OFFLINE=1
for tar_domain in ${domains[@]};
do
    for src_domain in  ${domains[@]};
    do
        if [ $src_domain != $tar_domain ];
        then
            echo "Train: $src_domain -> $tar_domain"
            python trainers.py --config config/train.yaml
        fi
    done
done