# JigsawNet
JigsawNet is an image fragment reassembly system. It is able to robustly reassemble irregular shredded image fragments.

This repository includes CNN-based pairwise alignment measurement and loop-closure based global reassembly. Please check our published paper for more algorithm details....

There are part of reassembly results from various public datasets and our own datasets (1st, 2nd and 3rd row contains 9, 100, 376 piece fragments repectively).

![demon](https://raw.githubusercontent.com/Lecanyu/JigsawNet/master/Examples/demo.png)





# 1. Prerequisites

We have tested this code on Windows10 x64 operation system.
We developed a CNN to measure pairwise compatibility, which has been implemented on Python 3.6 and tensorflow.
To globally reassemble, we designed a loop-based composition to calculate consistently reassembly result. The global algorithm has been implemented in C++ on Microsoft Visual Studio 2015 with CUDA 8 or 9 support. 

You should install below dependencies to run pairwise compatibility measurement part.
* Python 3.6
* Tensorflow 1.7.0 and its dependencies

You should install below libraries to run global reassembly part.
* OpenCV 3.4.1
* Eigen 3.3.4
* CUDA 8.0 or 9.0

Other version of those dependencies have not tested.

If you want to compile or run this code on different environments, a few of modifications will be needed.



# 2. Run pairwise compatibility measurement


# 3. Run global reassembly


# 4. Datasets and pre-trained net parameters
Our experiment datasets and pre-trained model can be downloaded [here](https://drive.google.com/open?id=1sUIcAzFTJNAAEEhqdYAKMKgzjVwRvsP4).

From this link, you can find 5 different datasets (one for training and four for testing) and the JigsawCNN parameters checkpoint which has been trained from the training dataset. 

You can directly load this checkpoint to run the example data. 

Note: For successfully load the checkpoint on your machine, you should modify the checkpoint file to correct path (i.e. JigsawCNN_checkpoint/g0/checkpoint, JigsawCNN_checkpoint/g1/checkpoint, ...). 
Since this code has been implemented on tensorflow, and the pretrained parameters can only be used on tensorflow library.






# 4. Citation
If this implementation is useful in your research, please cite XXX
