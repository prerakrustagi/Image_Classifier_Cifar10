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


  
