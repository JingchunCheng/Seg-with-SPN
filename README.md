Learning to Segment Instances in Videos with Spatial Propagation Network
=========================================
![alt text](http://vllab1.ucmerced.edu/~ytsai/CVPR17/cvpr17_workshop_git.png)

[This paper](http://davischallenge.org/challenge2017/papers/DAVIS-Challenge-6th-Team.pdf) is now available at the 2017 DAVIS-Challenge website.

Check our results in this [video](https://www.youtube.com/watch?v=JMCYk9w_TyA&feature=youtu.be).

# Cite the Paper
-------------------------------------------
If you find that our method is useful in your research, please cite:
```
@article{DAVIS2017-6th,
  author = {J. Cheng and S. Liu and Y.-H. Tsai and W.-C. Hung and S. Gupta and J. Gu and J. Kautz and S. Wang and M.-H. Yang}, 
  title = {Learning to Segment Instances in Videos with Spatial Propagation Network}, 
  journal = {The 2017 DAVIS Challenge on Video Object Segmentation - CVPR Workshops}, 
  year = {2017}
}
```
Contact: Jingchun Cheng (chengjingchun at gmail dot com)

# About the Code
-------------------------------------------
* The code released here mainly consistes of two parts in the paper: foreground segmentation and instance recognition.

* It contains the parent net of foreground segmentation and training codes for instance recognition networks.

* The **matlab_code** folder contains a simple version of our CRAF step for segmentation refinement.


# Installation Requirements
-------------------------------------------
Install `caffe` and `pycaffe` at http://caffe.berkeleyvision.org/.


# Installation
-----------------------------------------------------
1. Download the offline pre-trained foreground segmentation model.

`cd $Seg-with-SPN`

`cd ResNetF`

`wget https://www.dropbox.com/s/sifnbkgrvbzkttz/PN_ResNetF.caffemodel`

`mkdir models`

2. Download [DAVIS 2017 dataset](http://davischallenge.org/code.html) and put it in **Seg-with-SPN/data**.

3. Train per-object recognition model.

`cd $Seg-with-SPN/python_scripts`

`python solve.py ../ResNetF/PN_ResNetF.caffemodel ../ResNetF/testnet_per_obj/choreography/solver_1.prototxt`

*per-video foreground model can be trained in similar way.

4. Test your models.

`python infer_test_perobj.py model_iteration class_name object_id`

`e.g. python infer_test_perobj.py 3000 lions 2`

# Acknowledgement
--------------------------------------------------
Seg-with-SPN uses the following open source code:
* [ResNet-101](https://github.com/KaimingHe/deep-residual-networks) for initializing segmentation branch.


