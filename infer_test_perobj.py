import os,sys
sys.path.insert(0, "fcn_python/")
sys.path.insert(0, "python_layers/")

import caffe
import surgery

import numpy as np
from PIL import Image
import scipy.io

from scipy.misc import imresize

import os
from scipy import io

import shutil

import sys


def load_image(im_name, scale):
      # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(im_name)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    if im.height>H or im.width>W:
      in_ = imresize(in_, size=(H, W), interp="bilinear")
    if scale != 1:
      in_ = imresize(in_, size=(np.int(im.height * scale), np.int(im.width * scale)), interp="bilinear")
    in_ = in_.transpose((2,0,1))
    print >> sys.stderr, 'loading {}'.format(im_name)
    return in_



num_iter     = np.int(sys.argv[1])
cls_name     = sys.argv[2]
obj_id       = np.int(sys.argv[3])

scale        = 1

caffe_model  = 'ResNetF/models/{}_obj_{}_iter_{}.caffemodel'.format(cls_name,obj_id,num_iter)
deploy_proto = 'ResNetF/testnet_per_obj/deploy.prototxt'
file_out     = 'ResNetF_perobj_{}'.format(num_iter)


caffe.set_mode_gpu()
net = caffe.Net(deploy_proto , caffe_model, caffe.TEST)

davis_dir = 'data/DAVIS/'
res_dir   = 'results/'
split_f   = '{}/ImageSets/2017/test-challenge.txt'.format(davis_dir)
indices   = open(split_f, 'r').read().splitlines()


W = 854
H = 480

for idx in range(len(indices)):
    clip      = indices[idx]
    if clip != cls_name:
       continue

    print(clip)
    file_path = '{}/JPEGImages/480p/{}'.format(davis_dir, indices[idx])
    images    = os.listdir(file_path)

    if os.path.exists('{}/{}'.format(res_dir, file_out)) == False:
       os.mkdir('{}/{}'.format(res_dir, file_out))
    if os.path.exists('{}/{}/{}'.format(res_dir, file_out, clip)) == False:
       os.mkdir('{}/{}/{}'.format(res_dir, file_out, clip))

    for idx2 in range(len(images)):    
      im_name    = images[idx2]
      out_name   = '{}/{}/{}/{}_{}.png'.format(res_dir, file_out, clip, im_name[0:len(im_name)-4], obj_id)
      im_name    = '{}/{}'.format(file_path, images[idx2])

      img1 = load_image(im_name, scale)
    
      net.blobs['data'].reshape(1, *img1.shape) 
      net.blobs['data'].data[...] = img1

      net.forward()
      out = net.blobs['score_ori'].data[0].argmax(axis=0)
      out = out*255
      out = np.array(out, dtype=np.float32)
      res_img = Image.fromarray(out)
      out_name_1 = out_name[0:len(im_name)-4]
      out_name_1 = '{}.jpg'.format(out_name_1)
      res_img.convert('L').save(out_name_1)
      print(out_name_1)

      out = net.blobs['prob'].data[0,1,...]
      out = np.array(out, dtype=np.float32)
      out = np.uint8(out*255)
      res_img = Image.fromarray(out)
      if scale != 1:
         res_img = imresize(res_img, size=(H, W), interp="bilinear")
         res_img = Image.fromarray(res_img)
 
      res_img.convert('L').save(out_name)
      print(out_name)
   
print('done')




