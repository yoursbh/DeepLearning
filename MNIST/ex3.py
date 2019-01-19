
# TensorFlow execices: MNIST
# Hao BAI
# ref: https://www.jianshu.com/p/ceb06c568237

try:
    import sys
    import IPython  # to colorize traceback errors in terminal
    sys.excepthook = IPython.core.ultratb.ColorTB()
except:
    pass

from PIL import Image
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy as sp


#! Preprocessing Input Data ====================================================
#* load training data
train = pd.read_csv("./data/train.csv")
all_images = train.iloc[:,1:].values # image's pixel information:
                                     # size of 1 image = 28 * 28 pixels
                                     # 0 ≤ pixel-value ≤ 255
all_labels = train["label"].values.ravel() # image's label: the digit

#* pre-processes on image
all_images = all_images.astype(np.float) # convert pixel-value to float
all_images = np.multiply(all_images, 1.0/255.0) # convert pixel-value to decimal
                                                # number in interval [0,1]
print("Amount of input images: {}".format(all_images.shape[0]))
image_size = all_images.shape[1] # numbers of pixel in an image
print("|- Each image is a vector of length (size): {}".format(image_size))
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.int)
print("|- Each image has a size of: {}(width) x {}(height)".format(image_width, 
      image_height))
# x 占位符：用于稍后填充输入数据（image）
x = tf.placeholder("float", shape=[None, image_size], name="INPUT")
                  # "float" is equivalent to tf.float64 or tf.float32
                  # shape=[None, ...] means the 1st dimension (rows) will be
                  # auto allocated when feeding data (normally equals to the
                  # batch size)

#* pre-process on label
labels_count = np.unique(all_labels).shape[0] # numbers of category of labels:
                                              # here it's 10 since the digit is # 0 ~ 9
print("\nNumbers of category of label: {}".format(labels_count))
# y 占位符：用于稍后填充输出数据（label）
y = tf.placeholder("float", shape=[None, labels_count], name="OUTPUT")

#* one-hot encoding
labels = tf.one_hot(indices=all_labels, depth=labels_count, on_value=1,
                    off_value=0)
print("Amount of input image's labels: {}".format(labels.shape[0]))
print("|- Each label is a vector of length: {}".format(labels.shape[1]))
print("|- For example, the category '{}' is now {} after one-hot encoding"
      .format(all_labels[32], labels[32]))

#* separate input data into training set and validation set
VALID_SIZE = 2000 # numbers of data that will be used to valid model
train_images, train_labels = all_images[VALID_SIZE:], labels[VALID_SIZE:]
valid_images, valid_labels = all_images[:VALID_SIZE], labels[:VALID_SIZE]

BATCH_SIZE = 100 # divide training set in batches
n_batch = int(len(train_images)/BATCH_SIZE)
print("|- {} images will be used to train model through {} batches".format(
      len(train_images), n_batch))
print("|- {} images will be used to valid model".format(len(valid_images)))


#! Neural Network ==============================================================
def weight_variable(shape):
    """ Initialize the weights by normal distribution with standard deviation =
        0.1
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """ Initialize the biases by 0.1
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """ Computes a 2-D convolution and keep the same shape
        对图像进行2D卷积操作并保持输入图像的shape
    """
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    """ Max pooling 2x2最大值池化
    """
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

x_image = tf.reshape(x, [-1, 28, 28, 1]) # reshape the input data (images):
                                         # 2nd position: width of image
                                         # 3rd position: height of image
                                         # 4th position: channel of color

#* 1st convolution layer -------------------------------------------------------
W_conv1 = weight_variable([3,3,1,32]) # weight convolution: compute 32 
                                      # characterization per 3x3 patch
                                      # 1st and 2nd: size of patch
                                      # 3rd: number of input channels
                                      # 4th: number of output channels
b_conv1 = bias_variable([32]) # bias convolution: shape of bias should be the
                              # same as shape of output channels
""" Remark: 28x28图片卷积时步长为1，随意卷积后大小不变。然后按照2x2最大值池化，相当于从2x2
    块中提取一个最大值。所以第一次池化后大小为 [28/2,28/2] = [14,14]；第二次池化后大小为
    [14/2,14/2] = [7,7]
"""
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # convolution and ReLU
h_pool1 = max_pool_2x2(h_conv1) # pooling: image size is changed to 14x14 now

#* 2nd convolution layer -------------------------------------------------------
# compute 64 characterization
W_conv2 = weight_variable([6,6,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) # pooling: image is now changed to 7x7

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # transform pooling result in
                                                 # form of vector (1 dimension)

#* Fully connected layer -------------------------------------------------------
N_NEURON = 1024 # numbers of neuron in a layer of neural network
W_fc1 = weight_variable([7*7*64, N_NEURON])  # this layer has N_NEURON neurons
b_fc1 = bias_variable([N_NEURON])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# define Dropout operation
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# define output: 把1024个神经元的输入 变为 一个10维输出
W_fc2 = weight_variable([N_NEURON, labels_count])
b_fc2 = bias_variable([labels_count])
y_predict = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 # result (output)


#! Functions and Parametrage ===================================================
# loss function: 损失函数以交叉熵的平均值来衡量
loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict))

# gradient descent algorithm
# train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
train_step = tf.train.AdadeltaOptimizer(0.1).minimize(loss) # AdaDelta: an
                                                            # adaptive learning
                                                            # rate method

# calculate accuracy of prediction
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_predict,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# initialization
init = tf.global_variables_initializer()

# saving model
global_step = tf.Variable(0, name="Global_Step", trainable=False)
saver = tf.train.Saver()


#! Training ====================================================================
with tf.Session() as sess:
    sess.run(init) # initialize
    saver.restore(sess, "cnn.ckpt-BEST") # load saved model: choose the most 
                                         # accurate one
    # if use TensorFlow implemented one-hot encoding, we need to convert its
    # type to numpy.array
    train_labels = train_labels.eval()
    valid_labels = valid_labels.eval()
    
    # run mini batch SGD
    for epoch in range(1,21,1): # run for 20 loops
        # train model (neural network)
        for batch in range(n_batch):
            # get data batch by batch
            batch_x = train_images[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE]
            batch_y = train_labels[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE]
            # run trainning
            sess.run(train_step, feed_dict={x:batch_x, y:batch_y,
                     keep_prob:0.5})
    
        # show accuracy for every epoch
        loss_epoch = sess.run(loss, feed_dict={x: valid_images,
                                               y: valid_labels,
                                               keep_prob: 0.5})
        accuracy_epoch = sess.run(accuracy, feed_dict={x:valid_images,
                                                       y:valid_labels,
                                                       keep_prob: 0.5})
        print("Epoch {}: accuracy is {:.2f}%, loss is {}".format(epoch,
              accuracy_epoch*100, loss_epoch))
        # save model
        global_step.assign(epoch).eval()
        saver.save(sess, "./cnn.ckpt", global_step=global_step)


#! Tesing 1 single image =======================================================
def get_test_picture(filename):
    image_origin = Image.open(filename)
    width = 28 # width of destination image
    height = 28 # height of destination image
    image_resized = image_origin.resize((width, height), Image.ANTIALIAS)
    im_arr = np.array(image_resized.convert('L'), dtype=np.float32)
    nm = im_arr.reshape((1, width*height))
    return nm

def recognize_digit():
    print("\nRecognizing ...")
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "cnn.ckpt-BEST")  # load best model
        for i in [1,4,7,8]:
        # i = "unknown"
            testPicture = "{}.jpg".format(i) # filename of handwritten picture
            oneTestx = get_test_picture(testPicture) # load picture
            # get the prediction
            conv_y_predict = y_predict.eval(feed_dict={x: oneTestx, 
                                                       keep_prob:1.0})
            test_pred = np.argmax(conv_y_predict, axis=1)
            label_predict = np.int32(test_pred) # convert prediction number to
                                                # integer (one-hot decoding)
            print("|- The prediction of picture {}.jpg is: {}".format(i,
                  label_predict))

recognize_digit()
