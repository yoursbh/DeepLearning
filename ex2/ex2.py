
# TensorFlow execices: MNIST
# Hao BAI

try:
    import sys
    import IPython  # to colorize traceback errors in terminal
    sys.excepthook = IPython.core.ultratb.ColorTB()
except:
    pass

import tensorflow as tf
import pandas as pd
import numpy as np


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
one_hot_mode = 1
if one_hot_mode == 1: # use personal one-hot encoding
    def dense_to_one_hot(labels_dense, num_classes):
        """ ref: https://www.jqr.com/article/000243
            labels_dense: labels that will be operated
            num_classes: numbers of category of label
        """
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
        return labels_one_hot
    labels = dense_to_one_hot(all_labels, labels_count)
    labels = labels.astype(np.int)
elif one_hot_mode == 2: # use TensorFlow one-hot encoding
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
weights = tf.Variable(tf.zeros([image_size, labels_count])) # 权重
biases = tf.Variable(tf.zeros([labels_count])) # 偏移量
results = tf.add(tf.matmul(x,weights), biases) # 矩阵计算: x * weight + biases
#* activation function
predictions = tf.nn.softmax(results)
#* loss function = cost function
loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predictions))
#* gradient descent algorithm
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)


#! Initialization ==============================================================
init = tf.global_variables_initializer() # initialisation
#* calculate accuracy of prediction
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(predictions,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


#! Run session =================================================================
with tf.Session() as sess:
    sess.run(init) # initialize
    # if use TensorFlow implemented one-hot encoding, we need to convert its
    # type to numpy.array
    if one_hot_mode == 2:
        train_labels = train_labels.eval()
        valid_labels = valid_labels.eval()
    # run mini batch SGD
    for epoch in range(50): # run for 50 loops
        for batch in range(n_batch):
            # get data batch by batch
            batch_x = train_images[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
            batch_y = train_labels[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
            # train model
            sess.run(train_step, feed_dict={x:batch_x, y:batch_y})
            #* forward propagation is finished: means all samples (input data)
            #* have been used to train the model
    
        # show accuracy for every epoch
        loss_epoch = sess.run(loss, feed_dict={x: valid_images,
                                               y: valid_labels})
        accuracy_epoch = sess.run(accuracy, feed_dict={x:valid_images,
                                                       y:valid_labels})
        print("Epoch {}: accuracy is {:.2f}%, loss is {}".format((epoch+1),
              accuracy_epoch*100, loss_epoch))
        #* now begin backward propagation: means changing the weights


