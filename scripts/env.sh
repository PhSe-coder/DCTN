#!/bin/bash
conda create -n fdgr -y python==3.10
conda activate fdgr
conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.35.2
pip install lightning==2.1.2
pip install -i https://pypi.org/simple jsonargparse[signatures]==4.27.1
pip install matplotlib==3.8.2
pip install openpyxl==3.1.2
pip install SciencePlots==2.0.0
pip install -i https://pypi.org/simple stanza==1.7.0
pip install -i https://pypi.org/simple notebook==7.0.6
pip install fasttext==0.9.2
pip install pandas==2.1.3
pip install yapf==0.40.2
pip install -i https://pypi.org/simple tensorboard==2.15.1 tensorboardX==2.6.2.2
pip install nltk==3.8.1
pip install pytorch_revgrad==0.2.0
conda install -c dglteam/label/cu118 dgl==1.1.2.cu118
# pip install omegaconf==2.3.0