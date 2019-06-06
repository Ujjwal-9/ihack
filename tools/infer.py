#!/usr/bin/env python

import os
import time
import argparse

import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin


def parse_args(args):
    parser = argparse.ArgumentParser(description='convert model')
    parser.add_argument(
        '--img',
        help='path to image',
        type=str,
        required=True
    )
    parser.add_argument(
        '--xml',
        help='path to xml',
        type=str,
        required=True
    )
    parser.add_argument(
        '--bin',
        help='path to bin',
        type=str,
        required=True
    )
    parser.add_argument(
        '--plugin',
        help='path to plugin dir',
        type=str,
        required=False,
        default='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/'
    )
    parser.add_argument(
        '--cpu-ext',
        help='path to plugin dir',
        type=str,
        required=False,
        default='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
    )
    return parser.parse_args(args)


def main(args=None):
    args=parse_args(args)

    model_xml = args.xml
    model_bin = args.bin
    img_fn = args.img

    plugin = IEPlugin(device="CPU", plugin_dirs=args.plugin)
    plugin.add_cpu_extension(args.cpu_ext)
    net = IENetwork(model=model_xml, weights=model_bin)

    supported_layers = plugin.get_supported_layers(net)
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    print('not supported layers:', not_supported_layers)

    input_blob = 'input_1'
    cl_out_blob = 'classification/concat'
    rg_out_blob = 'regression/concat'

    print('out:', len(net.outputs))
    print('outputs', net.outputs)

    net.batch_size = 1
    n, c, h, w = net.inputs[input_blob].shape
    print(n, c, h, w)

    print("Loading model to the plugin")
    exec_net = plugin.load(network=net)
    del net

    # load images
    image = cv2.imread(img_fn)
    if image.shape[:-1] != (h, w):
        print("Image {} is resized from {} to {}".format(img_fn, image.shape[:-1], (h, w)))
        image = cv2.resize(image, (w, h))

    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    image = np.expand_dims(image, axis=0)

    res = exec_net.infer(inputs={input_blob: image})
    print(res)
    # classifications = res[cl_out_blob]
    # regressions = res[rg_out_blob]

    # print('cl', classifications.shape)
    # print('rg', regressions.shape)

    # print('cl', classifications)
    # print('rg', regressions)


if __name__ == '__main__':
    main()