#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.

a. Since the image classification problem we have is quite complicated, we found that a deep network is required. Thus,
we followed the structure of resnet18 that can be found at line 194, creating a model that consists of 4 “layers” that
are made up of two resblocks. A more complex resnet exceeded the size limit of 50MB. We tried other architectures such
as a simple convolution network, a modified version of vgg, and a densenet, but resnet performed the best.

b. Optimizer: Adam, it is fast and simple. The dataset we were provided with is considered to be large and had some
noise as some images were not categorised correctly and there were some unrelated images included. Adam is
computationally efficient, requires a small memory space, and is good at handling datasets with noise, which makes it
fit to the image classification problem we are facing.
Loss function: cross entropy since it is suited to image classification problems where there are more than two outputs.
It is used often for image classification of animals since it minimizes the difference between the predicted and actual
outputs.

c. During training we ran into a problem of overfitting where the train accuracy was quite larger than test accuracy.
Hence, we applied a series of image transformations to both the training and test datasets. For the training set, we
randomly cropped and resized the images to 224 x 224 to allow for more layers in the network. We also applied random
horizontal flips and random rotates, creating more input data for the net to learn from. For the training set, we simply
resized the images to 224 x 224 so as to match the input data. For both sets we then transformed the images to a tensor
and applied normalisation using the mean and standard deviation often used for ImageNet datasets. We observed
significant improvements as our model was able to reach 80% test accuracy with an increased ability to generalise.

d. Layer sizes: We attempted to lower the channel sizes to solve overfitting, but the model performed worse so we kept
them as is. By resizing the input images to 224 x 224, the output size before adaptive average pooling was
7 x 7. 224 x 224 performed the best out of the sizes we tried. 
Learning rate: we found that lr=0.001 (the default for Adam) performed the best, even over different architectures. 
Batch size: since a lower batch size generally leads to increased test accuracy, we used a batch size of 50 to train our
model
Train_val_split: we kept it at the default 0.8

e. We used weight decay (L2 regularization) of 0.0001 to prevent overfitting. Weight decay creates a new loss function
with original loss function and a L2 Norm of all the weight parameters of the model. We observed that the test accuracy
rose from 70% to 80%.
"""


############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
"""
Modify the input image to prevent overfitting.
mean and std are magical number found on internet and find them useful.

We based our transformations on the code from
https://towardsdatascience.com/improves-cnn-performance-by-applying-data-transformation-bf86b3f4cef4.
The mean and std are generally used for the ImageNet dataset.

RandomResizedCrop: randomly crop the image and resize to given size
RandomHorizontalFlip: randomly choose image to horizontally flip(left and right)
ToTensor: turn input image into data form
Normalize: normalize input data with given mean and std
"""
def transform(mode):
    # https://towardsdatascience.com/improves-cnn-performance-by-
    # applying-data-transformation-bf86b3f4cef4
    # ImageNet mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif mode == 'test':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
"""
The block of resnet which contains two convolutional layers with a skip connection which connects the original input
data to the output data.

parameters - 
in_channels: the number of channels of the input
out_channels: the number of channels of the output
stride: variables in convolutional layers, which is the distance the filter moves on the image data
downsample: the structure that is used to downsample the original input data to fit the output dat if it is the first
            block that concatenate input from another "layer".
"""
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        # bias are turned off since bias are added to affect the mean, which batch normalization is applied afterward
        # to remove the mean, hence it is no use to add bias.
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        last_out = x

        # first convolutional layer followed batch normalization and relu, which is the rectified linear unit function.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # second convolutional layer followed bu batch normalization
        x = self.conv2(x)
        x = self.bn2(x)

        # down sample if input data is from another "layer"
        if self.downsample is not None:
            last_out = self.downsample(last_out)

        # add the original input data to the output data (skip connection)
        x += last_out
        x = self.relu(x)

        return x


"""
The network that can be trained with input data and identify with new unknown data.
"""
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, 64, 64, 2)
        self.layer2 = self._make_layer(ResBlock, 64, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResBlock, 128, 256, 3, stride=2)
        self.layer4 = self._make_layer(ResBlock, 256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, 8)

    """
    make a "layer" with given number of resnet block
    """
    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        # add downsampling convolutional layer with batch normalization if the number of channels is changed between
        # input and output, or if the image size has changed between "layer"s
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                       nn.BatchNorm2d(out_channels))

        # construct "layer" with resnet blocks
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    # run through the resnet18
    def forward(self, input):
        # the first simple convolutional layer to process the input data and connect to 64 channels
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # the four "layers" to construct the resnet with number of blocks [2, 2, 2, 2], parameters are specified above
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # applies adaptive average pooling to the input image with size (7, 7) to (1, 1)
        x = self.avgpool(x)

        # flatten the input and classify them into 8 classes
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


net = Network()

############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    return


scheduler = None

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 50
epochs = 300