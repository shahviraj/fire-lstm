# fire-lstm
### Classification of Combustion instances into stable nad unstable combustions using ConvLSTM networks.

#### Please change the path of the dataset to the '../fire-dataset/combution_img_13.mat' in all of the above files before running the code. 

We submit this code as a part of our homework 4 submission for course ME592x at Iowa State University.

1) Pre-processing:

We load the images from the `.mat` file and resize (to 32x64 or similar resolution) it to a lower resolution. We then flatten it to store it as a vector.
We divide the data in Training and Testing datasets.

2) (i) LSTM model in Keras:

We first implement and try a simple LSTM network in Keras. We perform a 1D convolution on the data before passing it to LSTM block, so that the spatial information would be captured by our network. The code is available in `vanila-lstm.py`.

  (ii) Convolutional LSTM Network in Keras:

The file `convLSTM.py` contains our implementation of Convolutional LSTM Network in Keras. Please run the file to see the summary of our model, training accuracy and loss, and performance of our model on Test dataset.

3) Hyper-parameter Optimization:

We perform hyper-parameter optimization for two hyper-parameters, (i) `n_frames (strides)` and (ii) `n_filters`: number of filters used inside `ConvLSTM2D` layer of the Keras. Results are depicted in the report.

4) Different resolutions and aspect-ratios:

We train the models with different aspect ratios and resolutions for input images using the hyper-parameters obtained in part (3). Performance results are depicted in the report.

5) Depthwise separable convolutions:

We provide two implementations of the Depthwise Separable convolutional models. File `depthsep_conv.py` contains the higher capacity implementation using VGG architecture with residual connections. However, we figured out that such a large capacity network is not required for our dataset, so we provide a smaller architecture with less capacity in the file `depthsep_conv-small.py`. We observe the improvement in training times, speed and testing accuracy using depthwise separable convolutions. Performance results are depicted in the report.
