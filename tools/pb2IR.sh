#!/bin/bash

source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6 
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py  --input_model './tf_model/model_15.pb' --input_shape '[1,480,640,3]' --data_type FP32 --output_dir './inference' --tensorflow_use_custom_operations_config '/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/retinanet.json'
