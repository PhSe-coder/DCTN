#!/bin/bash
conda create -n fdgr -y python==3.10
conda activate fdgr
conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y -c dglteam/label/cu118 dgl==1.1.2.cu118
pip install transformers==4.35.2 lightning==2.1.2 matplotlib==3.8.2 openpyxl==3.1.2 SciencePlots==2.0.0 fasttext==0.9.2 pandas==2.1.3 yapf==0.40.2 nltk==3.8.1 pytorch_revgrad==0.2.0
pip install -i https://pypi.org/simple jsonargparse[signatures]==4.27.1 stanza==1.7.0 notebook==7.0.6 tensorboard==2.15.1 tensorboardX==2.6.2.2
pip uninstall psutil
# pip install omegaconf==2.3.0