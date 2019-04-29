#!/bin/bash

git clone https://github.com/fizyr/keras-retinanet.git ../keras-retinanet
git clone https://github.com/amir-abdi/keras_to_tensorflow.git
conda create -n retina python=3.6 -y
source activate retina
pip install numpy, xlrd, pandas
python generate.py
mkdir data/imgs/
cp data/images/0000009.jpg data/imgs/0000009.jpg
cd ../keras-retinanet
pip install .
python setup.py build_ext --inplace

echo "SETUP DONE"
