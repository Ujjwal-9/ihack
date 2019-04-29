#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import xml.etree.ElementTree as ET
import os, sys
from tqdm import tqdm


img_path = 'data/images/'
anno_path = 'data/annotations/'

train = pd.read_excel('./data/Training_set.xlsx')

test = pd.read_excel('./data/Test_set.xlsx')


a = []
for f in tqdm(range(len(train))):
    img_file = os.path.join(img_path, train.iloc[f, 1])
    tree = ET.parse(os.path.join(anno_path, train.iloc[f, 0]))
    root = tree.getroot()
    for obj in root.iter('object'):
        cls = obj.find('name').text
        xmlbox = obj.find('bndbox')
        bbox = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                int(xmlbox.find('ymax').text))
        l = [os.getcwd() + '/' + img_file, str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]),cls]
        a.append(l)

annotations = pd.DataFrame(a, columns=['image', 'xmin', 'ymix', 'xmax', 'ymax', 'class'])
annotations.to_csv("./data/annotations.csv", index=False, header=False)
cls = annotations['class'].unique()

class_ = pd.DataFrame(cls, columns=["class_name"])
class_['class_id'] = range(0,len(class_))
class_.to_csv('./data/class.csv', index=False, header=False)

cls_dict = {}
for i in range(len(class_)):
    cls_dict[class_.iloc[i, 1]] = class_.iloc[i, 0]

print(str(cls_dict))
