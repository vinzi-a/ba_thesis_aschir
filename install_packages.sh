#!/bin/bash
# here are not all packages but a lot of them. have a look in pip freeze to get additional package information. 
git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
cd PyTorch-YOLOv3/
pip3 install poetry --user
poetry install

poetry shell #activate virtual environment

#let's download the packages: 
pip3 install pytorchyolo --user

sudo apt-get install nvidia-cuda-toolkit 

#!/bin/bash

#git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
#cd PyTorch-YOLOv3/
pip3 install poetry --user
poetry install

poetry shell #activate virtual environment

#let's download the packages: 
pip3 install pytorchyolo --user

pip install opencv-python 

conda install matplotlib tqdmv imgaug tensorbooard terminaltables torchsummary natsort pandas Gputil

pip install tqdm #das noch

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install imgaug

conda install tensorbooard

pip install terminaltables

pip install torchsummary

pip install natsort

pip install pyexcel_xlsx

pip install pandas 

sudo apt install mesa-utils  # (Für Debian/Ubuntu-basierte Systeme) (read out ur Gpu version)

pip install Gputil