Learning to Segment Instances in Videos with Spatial Propagation Network
=========================================
![alt text](http://vllab1.ucmerced.edu/~ytsai/CVPR17/cvpr17_workshop_git.png)

[This paper](http://davischallenge.org/challenge2017/papers/DAVIS-Challenge-6th-Team.pdf) is now available at the 2017 DAVIS-Challenge website.

Check our results in this [video](https://www.youtube.com/watch?v=JMCYk9w_TyA&feature=youtu.be).

Contact: Jingchun Cheng (chengjingchun at gmail dot com)

# Cite the Paper
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
* The code released here mainly consistes of two parts in the paper: foreground segmentation and instance recognition.

* It contains the parent net for foreground segmentation and training codes for instance recognition networks.

* The **matlab_code** folder contains a simple version of our CRAF step for segmentation refinement.


# Installation Requirements
Install `caffe` and `pycaffe` at http://caffe.berkeleyvision.org/.


# Testing
* Download the [DAVIS 2017 dataset](http://davischallenge.org/code.html) and put it in the **data** folder.

* Download the offline pre-trained foreground segmentation model [here](http://vllab1.ucmerced.edu/~ytsai/CVPR17/PN_ResNetF.caffemodel) and put it in the **pretrained** folder.

* Test the general foreground/backgroung model. <br />
`cd $Seg-with-SPN/python_scripts` <br />
`python infer_test_fbbg.py PATH_OF_MODEL PATH_OF_RESULT VIDEO_NAME` <br />
`e.g. python infer_test_fbbg.py ../pretrained/PN_ResNetF.caffemodel ../results/fgbg lions`

* Test the object instance model. <br />
`cd $Seg-with-SPN/python_scripts` <br />
`python infer_test_perobj.py MODEL_ITERATION VIDEO_NAME OBJECT_ID` <br />
For example, on the 'lions' video for the 2nd object, run: <br />
`python infer_test_perobj.py 3000 lions 2`

* Run `example_CRAF.m` for a demo on CRAF segmentation refinement.

# Training
* Train the per-object recognition model. <br />
`cd $Seg-with-SPN/python_scripts` <br />
`python solve.py PATH_OF_MODEL PATH_OF_SOLVER` <br />
Foe example, on the 'choreography' video for the 1st object, run: <br />
`python solve.py ../pretrained/PN_ResNetF.caffemodel ../models/testnet_per_obj/choreography/solver_1.prototxt`

# Download Our Results
* General foreground/background segmentation
* Instance-level object segmentation without refinement
* Instance-level object segmentation with refinement
