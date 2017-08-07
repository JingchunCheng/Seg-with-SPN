import caffe
import numpy as np
import math
from scipy.misc import imresize
import sys

class LossWeightLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need rwo tops: score weight")


    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[0],bottom[0].data.shape[1],bottom[0].data.shape[2],bottom[0].data.shape[3])
        self.score  = np.zeros_like(bottom[0].data, dtype=np.float32) 
        self.weight = np.zeros_like(bottom[1].data, dtype=np.float32)

    def forward(self, bottom, top):
        self.score  = bottom[0].data 
        self.weight = bottom[1].data
        top[0].data[...] = self.score

    def backward(self, top, propagate_down, bottom):

        for i in range(2):
            if not propagate_down[i]:
                continue

            if i != 0:
                raise Exception('weight has no back prop')

            #A: it will at 1st iter, but later no
            bottom[0].diff[...] = np.zeros_like(bottom[0].diff, dtype=np.float32)
            for j in range(top[i].diff.shape[1]): 
                bottom[0].diff[0,j,...] = top[i].diff[0,j,...] * self.weight
                    




