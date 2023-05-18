#!/bin/bash
conda create -n venv -y python==3.10
conda activate venv
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install transformers==4.27.4
pip install lightning==2.0.0
pip install -i https://pypi.org/simple jsonargparse[signatures]==4.20.0
pip install yapf