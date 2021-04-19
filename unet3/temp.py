import csv
import json
import os
import numpy as np
import cv2
from matplotlib import pyplot


images_path = []
labels_path = []
csvFile = open('object.csv', "r")
reader = csv.reader(csvFile)
content = list(reader)
for item in content:
    images_path.append(item[0])
    labels_path.append(item[1])

print(len(images_path),len(labels_path),len(content))
print(images_path[0],labels_path[1])

