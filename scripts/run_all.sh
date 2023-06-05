#!/bin/bash
domains=('restaurant' 'service' 'laptop' 'device')

export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=1
for tar_domain in ${domains[@]};
do
    for src_domain in  ${domains[@]};
    do
        if [ $src_domain != $tar_domain ];
        then
            if [ $src_domain == 'laptop' -a  $tar_domain == 'device' ];
            then
                continue
            fi
            if [ $src_domain == 'device' -a  $tar_domain == 'laptop' ];
            then
                continue
            fi
            echo "Update config yaml files"
            python config/update.py --source $src_domain --target $tar_domain
            echo "Pretrain: $src_domain -> $tar_domain"
	        python trainers.py fit --config config/pretrain.yaml
            echo "Train: $src_domain -> $tar_domain"
            python trainers.py fit --config config/train.yaml
            model_path=/root/autodl-tmp/lightning_logs/*/checkpoints/fdgr-*.ckpt
            declare -a paths=($(ls -v $model_path)) # list all the model checkpoints
            echo "Test: $src_domain -> $tar_domain"
            python trainers.py test --config config/test.yaml --ckpt_path ${paths[-1]}
        fi
    done
done