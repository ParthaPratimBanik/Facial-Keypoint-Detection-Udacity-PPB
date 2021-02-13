# Facial-Keypoint-Detection-Udacity-PPB
Detecting 68 keypoints (coordinates (x,y)) of face on a dataset extracted from Youtube Faces Dataset


It is a facial keypoint detection project launched by udacity in Computer Vision Nanodegree program.

# Dataset
The images of the dataset has been extracted from the [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/), which includes videos of people in YouTube videos. In the dataset, ther are 3462 training images and 770 test images. The output is 68 keypoints (that means (x, y) coordinates) of face images.

# Reasoning to Build My CNN Model
I start with [this paper](https://arxiv.org/pdf/1710.00977.pdf), NaimishNet. They use 4 convolutional layers, increase filter size to the multiple of 2 of first layer filter size (32), and decrease filter size from 4x4 to 1x1 from first to last convolutional layer. Then I study [this paper](https://arxiv.org/pdf/1506.02897.pdf). I have found that they have found some benefit to use large kernels and use 8 and 5 (after spatial fusion) convolutional layers. For this reason, I have tried such CNN model where I can perform spatial fusion, and change the kernel size after fusing 2nd convolutional layer from 3x3 to 2x2 and 2x2 to 3x3. I use pooling layer while it is required to downsampling for decreasing the total number of trainable parameters. In total, 5 convolutional layers are used. The input image size for my CNN model is 90x90.

For avoid overfitting, the learning rate is set to 0.0001 of Adam optimizer (default is 0.001), the numbers of kernels are set to 32 and 64 for convolutional layers and only one fully-connected layer is used. It implies that the convolutional layer is not densed with large amount of kernels. It is conjectured that simple structure of the CNN model will avoid the overfitting tendency.
