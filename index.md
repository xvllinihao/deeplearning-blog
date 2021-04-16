# Spatially-sparse convolutional neural networks reproduction blog

This blog documents the reproduction efforts to the paper "Spatially-sparse convolutional neural networks" by Benjamin Graham. The reproduction project is conducted as part of the course Deep Learning course (CS4240) at TU Delft.

In this blog, we will briefly explain the architecture of the DeepCNet and the existing available code related to this paper ([link](https://github.com/btgraham/SparseConvNet)). As the existing code implements the spatially-sparse convolutional neural network with ResNet or VGG, we tried to make modifications to it to reproduce the DeepCNet as proposed in the paper. We applied the parameters used in the paper to see if the results can be reproduced, and we also adjusted the original parameters to see if there exists any performance improvement.

## Introduction

Convolutional neural networks are commonly used in deep learning tasks that involve data with a spatial or temporal structure, such as image classification. However, in some other tasks like handwritten character recognition, using conventional convolutional neural networks may be computationally inefficient, as they are used to learn features from traditional images, which is densely filled. The data for tasks like handwritten character recognition, nevertheless, is sparse. Spatially-spare convolutional neural networks can take advantage of this property, while still achieving the state-of-the-art performance.

## Model Architecture
The model described in the paper is called DeepCNet. Its main idea is to retain spatial information by applying slow max-pooling. It implements slow max-pooling by construct a deep CNN called DeepCNet and apply max-pooling after each convolution layer.

There are two parameters to describe the structure of DeepCNet (l,k): there are \(l+1\) layers with convolutional filters separated by $l$ layers of \(2\times 2\) max-pooling, and there are \(nk\) convolutional filters in the \(n^{th}\) layer. As for activation function, ReLU is used for hidden layers and softmax is used for the output layer. The filter size of the first layer is \(3 \times 3\), and for the rest of the layers it's \(2\times2\).

The figure below shows a DeepCNet with \(l=5\)​.

![l= 5](/img/l=5.png)



## Dataset

### Assamese_handwriting
A dataset of online handwritten assamese characters by collecting samples from 45 writers is created. Each writer contributed 52 basic characters, 10 numerals and 121 assamese conjunct consonants. The total number of entries corresponding to each writer is 183 (= 52 characters + 10 numerals + 121 conjunct consonants). The total number of samples in the dataset is 8235.

### CASIA
CASIA-OLHWDB1.1 dataset includes 3,755 GB2312-80 level-1 Chinese characters and 171 alphanumeric and symbols, written by 300 people. The total number of samples is 1174364.

## Code Implementation
2. The writer of this paper made [his implementation](https://github.com/facebookresearch/SparseConvNet/tree/master/sparseconvnet) open source . The existed implementation combined his later works like [Sparse 3D convolutional neural networks, BMVC 2015](http://arxiv.org/abs/1505.02890) and [Submanifold Sparse Convolutional Networks, 2017](https://arxiv.org/abs/1706.01307). Thus the performance of the network is boosted compared to the description in the paper. 

   Because the writer has combined his later work, the code's network architecture is different from that described in the paper. The main differences are:

   1. In the author's implementation, he used his self-developed submanifold convolution to replace the traditional convolution function. The main idea of submanifold convolution is that:

      > Due to the spatial sparsity, a site in a hidden layer is active if any of the areas in the layer that it takes as an input is active, and in each layer, we can learn a ground state, which is the same for all inactive sites. This can significantly save computational power, and it is implemented by calculating two matrices for each layer of the network during forward propagation: a feature matrix and a pointer matrix. The feature matrix consists of the feature vectors for each active spatial location in the layer and the ground state. The pointer matrix links each spatial location in the convolutional layer to the feature matrix, finding the corresponding features. 

   2. The author applies ResNet and VGGNet to network architecture to improve the performance of code.


## What we have done
Because we have limited time, our works are mainly based on the existed code implementation. 



### Data Augmentation
The existed work doesn't provide us with the data augmentation part. So we implement it by ourselves. In the paper, the writer introduces 3 type of data augmentation operation:1.translation 2. affine 3.adding histogram features.

The existed code can read Assamese_handwriting and Chinese_handwriting dataset as tensors. So we just do affine and translation operations on the tensors we get. This could be done by using PIL: first convert the tensors to PIL image objects, use `RandomAffine` method from `torchvison.transforms` to do translation and affine operations. Then convert the processed image objects back to tensors and feed them to the network. 

### Hyperparameter Tuning
We also tried to change the learning rate of the network to improve the performance of the network.


## Results
The experiment results are evaluated with top 1 error rate and top 5 error rate. We calculate this metric by the prediction matrix and the target matrix. For example, if we have 50 predictions to make for the test dataset and there are 183 classes in total, then the size of the prediction matrix is 50\*183, and the first column consists of the most likely class for each character in the test dataset. The target matrix is the same size, and the rows are filled with the true labels corresponding to each character in the test dataset. We then compare the two matrices and count how many rows have an overlap in the prediction and target matrix in the first 1 or 5 columns, and error rate is essentially this number divided by the size of test dataset. 


### Data Augmentation

| Dataset              | Data Augmentation | Traning Epoch | Top1 Error Rate | Top 5 Error Rate |
| -------------------- | ----------------- | ------------- | --------------- | ---------------- |
| Assamese_handwriting | None              | 100           | 1.94%           | 0.12%            |
| Assamese_handwriting | Affine            | 100           | 2.30%           | 0.30%            |
| Assamese_handwriting | Translation       | 100           | 2.45%           | 0.32%            |
| Chinese_handwriting  | None              | 100           | 1.70%           | 0.18%            |
| Chinese_handwriting  | Affine            | 100           | 1.85%           | 0.24%            |
| Chinese_handwriting  | Translation       | 100           | 1.95%           | 0.22%            |

### Hyperparameter Tuning

| Dataset              | Learning Rate | Training Epoch | Top1 Error Rate | Top 5 Error Rate |
| -------------------- | ------------- | -------------- | --------------- | ---------------- |
| Assamese_handwriting | 0.01          | 100            |                 |                  |
| Assamese_handwriting | 0.05          | 100            |                 |                  |
| Assamese_handwriting | 0.1           | 100            |                 |                  |
| Chinese_handwriting  | 0.01          | 100            |                 |                  |
| Chinese_handwriting  | 0.05          | 100            |                 |                  |
| Chinese_handwriting  | 0.1           | 100            |                 |                  |





### Confusion Matrix

The code can also generate a confusion matrix. Below is the confusion matrix of Assamese_handwriting dataset. Because it has 183 different classes, to visualize it clearly, we transfer the confusion matrix to a picture. In the picture, white points are point with the highest probabilities. As you can see the white points forms the diagonal of the matrix. This means, the predicted class is exactly the actual class. In another word, the test error is very low. 

![confusion matrix](/img/confusion-matrix.jpg)

## Reflection

### Li Xu
By doing this reproduction project, I am more familiar with the using of torch and have a better understanding about the knowledge we learn from lectures. And I also get to know the importance of writting enough comments in code. Because the authors' implementation has few comments, it is very difficult for me to understand the code and make some change or improvement. For example, because I can not understand the author's processing of the datasets, it is hard for me to implement the adding histogram features which is mentioned in the original paper.
### Xinyue Chen
This project gives me an opportuinty to know better the pytorch framework and how to make modifications to a convolutional neural network. Moreover, this paper and the reproduction project raised my awareness to place more consideration on computation efficiency when doing deep learning tasks. Although I can get an overall idea of the code implementation, I cannot comprehend some part of it still, such as the Sparse2Dense function due to a lack of comments, making it hard to explain the whole logic of the code. 

## Acknowledgement
Thanks Nergis and Xin for their kind help and suggestions.
