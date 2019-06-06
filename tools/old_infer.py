import glob

import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin

# ONLY one images!

fns = glob.glob('data/imgs/*jpg')

model_xml = './Inference/model_15.xml'
model_bin = './Inference/model_15.bin'

plugin = IEPlugin(device="CPU", plugin_dirs='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/')
plugin.add_cpu_extension('/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so')
net = IENetwork(model=model_xml, weights=model_bin)

supported_layers = plugin.get_supported_layers(net)
not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
print('not supported layers:', not_supported_layers)

input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

print('out:', len(net.outputs))

net.batch_size = len(fns)
n, c, h, w = net.inputs[input_blob].shape
print(n, c, h, w)

print("Loading model to the plugin")
exec_net = plugin.load(network=net, num_requests=2)
del net

# load images
for idx,fn in enumerate(fns):
    image = cv2.imread(fn)
    if image.shape[:-1] != (h, w):
        print("Image {} is resized from {} to {}".format(fn, image.shape[:-1], (h, w)))
        image = cv2.resize(image, (w, h))
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float64)

res = exec_net.infer(inputs={input_blob: image})
res = res[out_blob]

print(res.shape)
print("*"*50)
print(res)
print("*"*50)