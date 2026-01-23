#!/bin/bash

apt update 
pip install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
pip install submitit
# torch==1.8.1
# torchvision==0.9.1
pip install faiss-gpu==1.7.2
pip install opencv-python==4.6.0.66
pip install scikit-image==0.19.2
pip install scikit-learn==1.1.1
pip install shapely==1.8.2
pip install timm==0.5.4
pip install pyyaml==6.0
pip install colored
pip install fvcore==0.1.5.post20220512
pip install gdown==4.5.4

pip install transformers timm 
pip install --upgrade transformers timm

pip install git+https://github.com/lucasb-eyer/pydensecrf.git
#cd detectron2
#pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

apt -y install libfuse2
cd ../mmdetection-dev-2.x
pip install -e .

cd ../detectron2
pip install -e .
cd ../datapipefs
pip install -e .

pip install mmcv-full==1.7.0
#cd ../mmcv
#pip install -e .
#video part
cd ../CutLER
pip install -r videocutler/requirements.txt


