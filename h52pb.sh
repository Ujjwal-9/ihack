#!/bin/bash


read -p "Enter Your Model Name: "  model_name
source activate retina
python keras_to_tensorflow/keras_to_tensorflow.py --input_model="model/$model_name.h5" --output_model="tf_model/$model_name.pb"
echo "DONE"

