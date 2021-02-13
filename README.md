# Facial-Keypoint-Detection-Udacity-PPB
It is a facial keypoint detection project launched by udacity in Computer Vision Nanodegree program. Detecting 68 keypoints (coordinates (x,y)) of face on a dataset extracted from Youtube Faces Dataset.

# Dataset
The images of the dataset has been extracted from the [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/), which includes videos of people in YouTube videos. In the dataset, ther are 3462 training images and 770 test images. The output is 68 keypoints (that means (x, y) coordinates) of face images.

# Reasoning to Build My CNN Model
I start with [this paper](https://arxiv.org/pdf/1710.00977.pdf), NaimishNet. They use 4 convolutional layers, increase filter size to the multiple of 2 of first layer filter size (32), and decrease filter size from 4x4 to 1x1 from first to last convolutional layer. Then I study [this paper](https://arxiv.org/pdf/1506.02897.pdf). I have found that they have found some benefit to use large kernels and use 8 and 5 (after spatial fusion) convolutional layers. For this reason, I have tried such CNN model where I can perform spatial fusion, and change the kernel size after fusing 2nd convolutional layer from 3x3 to 2x2 and 2x2 to 3x3. I use pooling layer while it is required to downsampling for decreasing the total number of trainable parameters. In total, 5 convolutional layers are used. The input image size for my CNN model is 90x90.

For avoid overfitting, the learning rate is set to 0.0001 of Adam optimizer (default is 0.001), the numbers of kernels are set to 32 and 64 for convolutional layers and only one fully-connected layer is used. It implies that the convolutional layer is not densed with large amount of kernels. It is conjectured that simple structure of the CNN model will avoid the overfitting tendency.

# Optimizer & Loss Function Selection
Generally, for regression type problem, L2Loss, known as mean square error (MSE) loss is widely used. [This paper](https://arxiv.org/pdf/1711.06753.pdf) mentions it clearly. Besides, they had found that L1Loss and SmoothL1Loss are better than widely used L2loss (MSELoss). For this reason, I check the training losses over batches after training 50 epochs and select the best performed loss function. For my model, it is SmoothL1Loss.

# Number of Epoch and Batch Size Selection
The model is trained for 50 epochs and is checked the expected training loss level and also visualized the predicted keypoints of the images having high loss (>0.01). After several times of the experiments, it is found that, the model trains enough well within 50 epochs. In most cases, the training loss drops down from 0.5 to 0.007 in first 2 epochs, then it takes 25 more epochs to drop down to 0.0038. Finally, 16 more epochs are taked to drop down to 0.0025. For training, i have used minibatch training. The size of minibatch is 64. Experimentally, I have found that lower minibatch (from 4 to 32) is quite noisy and the training tends to overfitted. So, the batch size is set to 64.

# Visualize Layers of Trained CNN Model
Here, the filters of 1st convolutional layer of the trained CNN model is checked. Most of the filters emphaize to detect edges and they have a photonegative effect on the image where white pixels are darkened and dark pixels are whitened. Few of them creats blur effect, like gaussian filter. Few of them detects detects the texture of backgrounds.