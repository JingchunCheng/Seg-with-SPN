import caffe

import numpy as np
from PIL import Image

import cv2
from scipy.misc import imresize
from scipy.misc import imrotate

import sys

import random

# generate fg-bg label for object obj_id in frm1
class DAVIS1stFrmSpecificObjLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):

        # config
        params = eval(self.param_str)
        self.davis_dir = params['davis_dir']
        self.split     = params['split']
        self.mean      = np.array(params['mean'])
        self.random    = params.get('randomize', True)
        self.seed      = params.get('seed', None)
        self.scale     = params.get('scale', 1)
        self.augment    = params.get('with_augmentation', True)
        self.aug_params = np.array(params['aug_params']) #( aug_num, max_scale, max_rotate, max_translation, flip)
        self.H         = 480
        self.W         = 854
        self.obj_id    = params.get('obj_id')


        # two tops: data and label
        if len(top) != 3:
            raise Exception("Need to define three tops: data label, weight")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        self.classes    = ['']
        self.classes[0] = self.split
        self.idx        = 0
  

        if self.augment:
           self.aug_num         = np.int(self.aug_params[0])
           self.max_scale       = self.aug_params[1]
           self.max_rotate      = self.aug_params[2]
           self.max_transW      = self.aug_params[3]
           self.max_transH      = self.aug_params[4]
           self.flip            = (self.aug_params[5]>0)


    def reshape(self, bottom, top):

        while True:
            idx = self.idx

            if self.augment == False or random.randint(0, self.aug_num) == 0:
               self.img    = self.load_image(self.classes[self.idx])
               self.label  = self.load_label(self.classes[self.idx])
               self.img    = imresize(self.img,    size=(self.H, self.W), interp="bilinear")
               self.label  = imresize(self.label,  size=(self.H, self.W), interp="nearest")
            else:
               scale       =  (random.random()*2-1) * self.max_scale
               rotation    =  (random.random()*2-1) * self.max_rotate
               trans_w     =  np.int( (random.random()*2-1) * self.max_transW * self.W )
               trans_h     =  np.int( (random.random()*2-1) * self.max_transH * self.H )
               if self.flip:
                  flip     = (random.randint(0,1) > 0)
               else:
                  flip     = False
               self.img    = self.load_image_transform(self.classes[self.idx], scale, rotation, trans_h, trans_w, flip)
               self.label  = self.load_label_transform(self.classes[self.idx], scale, rotation, trans_h, trans_w, flip)


            if self.scale != 1:
               self.img   = imresize(self.img,    size=(np.int(self.H*self.scale), np.int(self.W*self.scale)), interp="bilinear")
               self.label = imresize(self.label,  size=(np.int(self.H*self.scale), np.int(self.W*self.scale)), interp="nearest")

            self.weight = self.calculate_weight(self.label)

            self.img   = self.img.transpose((2,0,1))
            self.label = self.label[np.newaxis, ...]
            print >> sys.stderr, self.img.shape
            break            

        # reshape tops to fit (leading 2 is for batch dimension)
        top[0].reshape(1, *self.img.shape)
        top[1].reshape(1, *self.label.shape)
        top[2].reshape(1, *self.weight.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.img
        top[1].data[...] = self.label
        top[2].data[...] = self.weight

    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        print >> sys.stderr, 'loading Original {}/00000.jpg'.format(idx)
        im = Image.open('{}/JPEGImages/480p/{}/00000.jpg'.format(self.davis_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}/Annotations/2017/{}/00000.png'.format(self.davis_dir, idx))
        label = np.array(im, dtype=np.uint8)
        label = np.uint8(label==self.obj_id)
        print >> sys.stderr, 'Number of Objects: {}'.format(np.max(label))
        return label


    def calculate_weight(self, label):
       weight    = np.zeros_like(label, dtype = np.float32) 
       num_class = np.max(label) + 1
       for class_id in range(num_class):
           pos        = np.where(label == class_id)
           weight_idx = np.float32(1 - len(pos[0])*1.0/weight.size)
 #          print(weight_idx)
           for idx  in range (len(pos[0])):
               weight[pos[0][idx], pos[1][idx]] = weight_idx

       return weight


    def load_image_transform(self, idx, scale, rotation, trans_h, trans_w, flip):
       img_W = np.int( self.W*(1.0 + scale) )
       img_H = np.int( self.H*(1.0 + scale) ) 

       print >> sys.stderr, 'loading {}/00000.jpg'.format(idx)
       print >> sys.stderr, 'scale: {}; rotation: {}; translation: ({},{}); flip: {}.'.format(scale, rotation, trans_w, trans_h, flip)

       im = Image.open('{}/JPEGImages/480p/{}/00000.jpg'.format(self.davis_dir, idx))
       im    = im.resize((img_W,img_H))
       im    = im.transform((img_W,img_H),Image.AFFINE,(1,0,trans_w,0,1,trans_h))
       im    = im.rotate(rotation)
       if flip:
          im = im.transpose(Image.FLIP_LEFT_RIGHT)
       
       if scale>0:
          box = (np.int((img_W - self.W)/2), np.int((img_H - self.H)/2), np.int((img_W - self.W)/2)+self.W, np.int((img_H - self.H)/2)+self.H)
          im  = im.crop(box)
       else:
          im  = im.resize((self.W, self.H))
       
 #      im_name = 'img_{}_{}.jpg'.format(trans_h,trans_w)
 #      im.save(im_name,"JPEG")
 #      print(im.size)

       in_ = np.array(im, dtype=np.float32)
       in_ = in_[:,:,::-1]
       in_ -= self.mean  

       return in_


    def load_label_transform(self, idx, scale, rotation, trans_h, trans_w, flip):
        img_W = np.int( self.W*(1.0 + scale) )
        img_H = np.int( self.H*(1.0 + scale) )
        
#        print >> sys.stderr, 'loading {}'.format(idx)
#        print >> sys.stderr, 'scale: {}; rotation: {}; translation: ({},{}); flip: {}.'.format(scale, rotation, trans_w, trans_h, flip)

        im = Image.open('{}/Annotations/2017/{}/00000.png'.format(self.davis_dir, idx))
        im    = im.resize((img_W,img_H))
        im    = im.transform((img_W,img_H),Image.AFFINE,(1,0,trans_w,0,1,trans_h))
        im    = im.rotate(rotation)
        if flip:
           im = im.transpose(Image.FLIP_LEFT_RIGHT)

        if scale>0:
           w_start = np.int(random.random()*(img_W - self.W))
           h_start = np.int(random.random()*(img_H - self.H))
           box     = (w_start, h_start, w_start+self.W, h_start+self.H)
           im      = im.crop(box)
        else:
           im  = im.resize((self.W, self.H))

#        im_name = 'label_{}_{}.png'.format(trans_h,trans_w)
#        im.save(im_name,"PNG")
#        print(im.size)
        label = np.array(im, dtype=np.uint8)
        label = np.uint8(label==self.obj_id)
#        print(label)
        print >> sys.stderr, 'Number of Objects: {}'.format(np.max(label))
        
        return label




