# ML_Project | Image Classifier based on CIFAR-10 Dataset

## List of files
* data_batch_1
* data_batch_2
* data_batch_3
* data_batch_4
* data_batch_5

Each file packs the data using pickle module in python.

## Understanding the original image dataset
* The original one batch data is (10000 x 3072) matrix expressed in numpy array.
* The number of columns, (10000), indicates the number of sample data.
* The row vector, (3072) represents an color image of 32x32 pixels.
* Since this project is going to use CNN for the classification tasks, the original row vector is not appropriate. In order to feed an image data into a CNN model, the dimension of the input tensor should be either (width x height x num_channel) or (num_channel x width x height).

We are going forward with (width x height x num_channel)

* Each image vector is of size 3072 which is equal to 32x32x3
* In order to get the input tensor shape for the CNN, we need to use the following
  * Reshape - convert [1, 3072] to [3, 32, 32]
  * Transpose - convert [3, 32, 32] to [32, 32, 3]

![Image Pre-process Steps](Images/Image_Preprocess.png)
  
## The original labels
The label data is just a list of 10,000 numbers ranging from 0 to 9, which corresponds to each of the 10 classes in CIFAR-10
* airplane : 0
* automobile : 1
* bird : 2
* cat : 3
* deer : 4
* dog : 5
* frog : 6
* horse : 7
* ship : 8
* truck : 9

## Pre-Process Input Data-Set
The pixel values ranges from 0 to 255. When such a value is passed into sigmoid function, the output is almost always 1, and when it is passed into ReLU function, the output could be very huge. When back-propagation process is performed to optimize the network, these output values could lead to an vanishing gradient problems. In order to avoid the issue, it is better let all the values be around 0 and 1.

### Solution
Min-Max Normalization

```
  Normalized_Value = (value - min_value) / (max_value - min_value)
```

