import os
import sys
import numpy as np
import matplotlib.pyplot as plt

#caffe_root = '/work/pack/caffe/'
#sys.path.insert(0, caffe_root + 'python')
import caffe

MODEL_FILE = '../lenet.prototxt'
PRETRAINED = '../lenet_iter_10000.caffemodel'


#图片像素28*28  
#单通道BMP格式

IMAGE_FILE = './1.bmp'
input_image = caffe.io.load_image(IMAGE_FILE, color=False)
print(input_image)

#print input_image
net = caffe.Classifier(MODEL_FILE, PRETRAINED)
#predict 预测
prediction = net.predict([input_image], oversample = False)
caffe.set_mode_cpu()
print ('predicted class:', prediction[0].argmax())
