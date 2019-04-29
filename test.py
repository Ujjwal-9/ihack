#!/usr/bin/env python
# coding: utf-8

# Test Info
model_version = '15'
img_name = '2018-03-14_10-31-53-311527_leftImg8bit'

import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

keras.backend.tensorflow_backend.set_session(get_session())


# ## Load RetinaNet model
model_path = os.path.join('.', 'model', 'model_{}.h5'.format(model_version))
model = models.load_model(model_path, backbone_name='resnet50')


# load label to names mapping for visualization purposes
labels_to_names = {0: 'vehicle fallback',
 1: 'bus',
 2: 'car',
 3: 'truck',
 4: 'motorcycle',
 5: 'autorickshaw',
 6: 'rider',
 7: 'person',
 8: 'traffic light',
 9: 'traffic sign',
 10: 'animal',
 11: 'bicycle',
 12: 'caravan'}


# ## Run detection on example

image = read_image_bgr('./data/images/{}.jpg'.format(img_name))

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break
        
    color = label_color(label)
    
    b = box.astype(int)
    draw_box(draw, b, color=color)
    
    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)
    print(caption)
    
plt.figure(figsize=(50, 50))                          
plt.axis('off')
plt.imshow(draw)
plt.savefig('./data/Tests/{}_{}.png'.format(img_name, model_version), bbox_inches='tight')
plt.show()

