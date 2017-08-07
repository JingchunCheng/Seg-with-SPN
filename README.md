Learning to Segment Instances in Videos with Spatial Propagation Network
=========================================
Introduction
-----------------------------------------
We propose a deep learning-based framework for instance-level object segmentation. Our method mainly consists of three steps. First, We train a generic model based on ResNet-101 for foreground/background segmentations. Second, based on this generic model, we fine-tune it to learn instance-level models and segment individual objects by using augmented object annotations in first frames of test videos. To distinguish different instances in the same video, we compute a pixel-level score map for each object from these instance-level models. Each score map indicates the objectness likelihood and is only computed within the foreground mask obtained in the first step. To further refine this per frame score map, we learn a spatial propagation network. This network aims to learn how to propagate a coarse segmentation mask spatially based on the pairwise similarities in each frame. In addition, we apply a filter on the refined score map that aims to recognize the best connected region using spatial and temporal consistencies in the video. Finally, we decide the instance-level object segmentation in each video by comparing score maps of different instances.

[This paper](http://davischallenge.org/challenge2017/papers/DAVIS-Challenge-6th-Team.pdf) is now available at the 2017 DAVIS-Challenge website.

Video Result the Proposed Method
-------------------------------------------
[Final Result](https://www.youtube.com/watch?v=JMCYk9w_TyA&feature=youtu.be)


About the Code
-------------------------------------------
The code released here is for foreground segmentation and instance recognition step.


Requirements
-------------------------------------------
Requirements for `caffe` and `pycaffe`.


Installation
-----------------------------------------------------
1. Download offline trained foreground segmentation model.

`cd $Seg-with-SPN`

`cd ResNetF`

`wget https://www.dropbox.com/s/sifnbkgrvbzkttz/PN_ResNetF.caffemodel`

`mkdir models`

2. Download DAVIS 2017 dataset and put it in `$Seg-with-SPN/data`.

3. Install caffe and pycaffe.

4. Train per-object recognition model.

`cd $Seg-with-SPN/python_scripts`

`python solve.py ../ResNetF/PN_ResNetF.caffemodel ../ResNetF/testnet_per_obj/choreography/solver_1.prototxt`

*per-video foreground model can be trained in similar way.

5. Test your models.
`python infer_test_perobj.py model_iteration class_name object_id`

`e.g. python infer_test_perobj.py 3000 lions 2`




Acknowledgement
--------------------------------------------------
Seg-with-SPN uses the following open source code:
* [ResNet-101](https://github.com/KaimingHe/deep-residual-networks) for initializing segmentation branch.


