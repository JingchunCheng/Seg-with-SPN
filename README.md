Learning to Segment Instances in Videos with Spatial Propagation Network
=========================================
![alt text](http://vllab1.ucmerced.edu/~ytsai/CVPR17/cvpr17_workshop_git.png)

[This paper](http://davischallenge.org/challenge2017/papers/DAVIS-Challenge-6th-Team.pdf) is now available at the 2017 DAVIS-Challenge website.

Check our results in this [video](https://www.youtube.com/watch?v=JMCYk9w_TyA&feature=youtu.be).

Contact: Jingchun Cheng (chengjingchun at gmail dot com)

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

# About the Code
-------------------------------------------
* The code released here mainly consistes of two parts in the paper: foreground segmentation and instance recognition.

* It contains the parent net for foreground segmentation and training codes for instance recognition networks.

* The **matlab_code** folder contains a simple version of our CRAF step for segmentation refinement.


# Installation Requirements
-------------------------------------------
Install `caffe` and `pycaffe` at http://caffe.berkeleyvision.org/.


# Demo
-----------------------------------------------------
* Download the [DAVIS 2017 dataset](http://davischallenge.org/code.html) and put it in the **data** folder.

* Download the offline pre-trained foreground segmentation model [here](https://www.dropbox.com/s/sifnbkgrvbzkttz/PN_ResNetF.caffemodel) and put it in the **pretrained** folder.

* Train the per-object recognition model.

`cd $Seg-with-SPN/python_scripts`

`python solve.py ../pretrained/PN_ResNetF.caffemodel ../models/testnet_per_obj/choreography/solver_1.prototxt`

* Test the models.

`python infer_test_perobj.py model_iteration class_name object_id`

`e.g. python infer_test_perobj.py 3000 lions 2`

# Acknowledgement
--------------------------------------------------
Seg-with-SPN uses the following open source code:
* [ResNet-101](https://github.com/KaimingHe/deep-residual-networks) for initializing segmentation branch.


