# STNN-code
STNN uses recurrent convolutional network to learn temporal dependencies, and uses densely connected convolutional network to learn spatial dependencies and handle spatial sparsity. Experiments were conducted in Wuhan, China, where taxicab trajectory data were used to train and validate STNN.

Dependencies
-----

* python (python2 is recommended)
* theano (1.0.4)
* keras (2.1.1) (modified version should be [download](https://github.com/JasonYanxx/STNN/blob/master/site-packages/keras-211.rar))
* h5py
* numpy
* spicy
* pandas
* CUDA 8.0 or later version
* cuDNN 7.0.5

Usage
---------
* ```python Exp_WuHan.py train```
* ```python Exp_WuHan.py predict```

Dataset Description
-------
* Download:
	* Link: [STNN_WuHanTaxi_10min_192x192_R3_D3_W3_N1_Y2_2D.h5](http://ggssc.whu.edu.cn/ggssc/downloads/STNN/STNN_WuHanTaxi_10min_192x192_R3_D3_W3_N1_Y2_2D.rar)
	* The data are organized using hdf5 format,and you can use `h5py` to manipulate the data.
	* Once you download the data,you need to place it in `...\STNN\Data\CACHE`.

* Description

The data contain training and testing data, both of them are composed of traffic flow matrix and meta data(date information).The training data contains 1727 samples while the testing contains 1296 samples. The main parts of the data are shown below:

```
# trian data,
X_train_0 : a 5D tensor of shape (1727,3,1,192,192) 
X_train_1 : a 4D tensor of shape (1727,3,192,192)
X_train_2 : a 4D tensor of shape (1727,3,192,192)
X_train_3 : a 2D tensor of shape (1727,8)
Y_train: a 4D tensor of shape (1727,1,192,192)
T_train: a 1D tensor of shape (1727,)
# test data
X_test_0 : a 5D tensor of shape (1296,3,1,192,192)
X_test_1 : a 4D tensor of shape (1296,3,192,192)
X_test_2 : a 4D tensor of shape (1296,3,192,192)
X_test_3 : a 2D tensor of shape (1296,8)
Y_test: a 4D tensor of shape (1296,1,192,192)
T_test: a 1D tensor of shape (1296,)

```
For example:

* `X_train_0[0][2][0]` is a `192x192` traffic flow matrix which is used for the input of `Recent part`
* `X_train_1[0][2]` is a `192x192` traffic flow matrix which is used for the input of `Daily part`
* `X_train_2[0][2]` is a `192x192` traffic flow matrix which is used for the input of `Weekly part`
* `X_train_3[0]=00100001` represents `Wendsday, Weekday`,which is used for the input of `External part`
* `T_train[0]=20150415002` represents `2015/04/15 00:10:00`,which is the timestamp accosiated with predict traffic flow matrix

Visualization
-----------

<table>
    <tr>
        <td ><center>Traffic flow matrix on 2015/04/15 09:10:00<img src=Figure_1.png width="400" height="400"></center></td>
        <td ><center>Traffic flow on road network on 2015/04/15 09:10:00<img src=Figure_2.png width="400" height="400"></center></td>
    </tr>
</table>
