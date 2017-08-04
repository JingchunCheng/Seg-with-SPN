Learning to Segment Instances in Videos with Spatial Propagation Network
=========================================
Introduction
-----------------------------------------
We propose a deep learning-based framework for instance-level object segmentation. Our method mainly consists of three steps. First, We train a generic model based on ResNet-101 for foreground/background segmentations. Second, based on this generic model, we fine-tune it to learn instance-level models and segment individual objects by using augmented object annotations in first frames of test videos. To distinguish different instances in the same video, we compute a pixel-level score map for each object from these instance-level models. Each score map indicates the objectness likelihood and is only computed within the foreground mask obtained in the first step. To further refine this per frame score map, we learn a spatial propagation network. This network aims to learn how to propagate a coarse segmentation mask spatially based on the pairwise similarities in each frame. In addition, we apply a filter on the refined score map that aims to recognize the best connected region using spatial and temporal consistencies in the video. Finally, we decide the instance-level object segmentation in each video by comparing score maps of different instances.

[This paper](http://davischallenge.org/challenge2017/papers/DAVIS-Challenge-6th-Team.pdf) is now available at the 2017 DAVIS-Challenge website.

Video Result the Proposed Method
-------------------------------------------
[Final Result]()


Code
-------------------------------------------
