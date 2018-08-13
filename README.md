# JigsawCNN
JigsawCNN is an image fragment reassembly system. It is able to robustly reassemble irregular shredded image fragments.

Please check our published paper for more details....


# 1. Prerequisites

We have tested this code on Windows10 x64 operation system.
The JigsawCNN part has been implemented on Python 3.6 and tensorflow. 
The global reassembly part has been implemented on Microsoft Visual Studio 2015 with CUDA 8 or 9 support. 

If you want to compile or run this code on different development environment, a few of modifications will be needed.

The following install instruction only works on Windows10 X64 OS and Microsoft Visual Studio 2013 (x86 solution).

For your convenience, the 3rd dependencies can be downloaded [here](https://drive.google.com/open?id=1mQyDLpdVoQKWVBcRU9iTdNWRnw1liQKY).


Modify environment variables
--------------------
After Install/unzip all of dependencies. You should create some environment variables and specify correct path. Please refer "env.txt" for more details.


# 2. Build
When you finish all of environment configuration, you can use VS2013 to build the whole project. If all goes well, you can run it without any errors.


# 3. Datasets and trained net parameters
Our experiment datasets can be downloaded [here](https://drive.google.com/open?id=1sUIcAzFTJNAAEEhqdYAKMKgzjVwRvsP4).

From this link, you can find 5 different datasets (one for training and four for testing) and the JigsawCNN parameters checkpoint which has been trained from the training dataset. 

You can directly load this checkpoint to run the example data. 

Note: For successfully load the checkpoint on your machine, you should modify the checkpoint file to correct path (i.e. JigsawCNN_checkpoint/g0/checkpoint, JigsawCNN_checkpoint/g1/checkpoint, ...). 
Since this code has been implemented on tensorflow, and the pretrained parameters can only be used on tensorflow net.






# 4. Citation
If this implementation is useful in your research, please cite XXX
