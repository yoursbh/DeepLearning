
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
#* saved file
global_step = tf.Variable(0, name='global_step', trainable=False)
saver = tf.train.Saver()


#! Train model =================================================================
with tf.Session() as sess:
    sess.run(init) # initialize
    saver.restore(sess, "model.ckpt-19")  # load model if necessary
    # if use TensorFlow implemented one-hot encoding, we need to convert its
    # type to numpy.array
    if one_hot_mode == 2:
        train_labels = train_labels.eval()
        valid_labels = valid_labels.eval()
    
    # run mini batch SGD
    for epoch in range(20): # run for 50 loops
        #* at the end of each epoch, it begins backward propagation (means 
        #* changing the weights)
        for batch in range(n_batch):
            #* every batch means a forward propagation is finished (all samples 
            #* have been used to train the model)
            # get data batch by batch
            batch_x = train_images[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
            batch_y = train_labels[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
            # train model
            sess.run(train_step, feed_dict={x:batch_x, y:batch_y})
    
        # show accuracy for every epoch
        loss_epoch = sess.run(loss, feed_dict={x: valid_images,
                                               y: valid_labels})
        accuracy_epoch = sess.run(accuracy, feed_dict={x:valid_images,
                                                       y:valid_labels})
        print("Epoch {}: accuracy is {:.2f}%, loss is {}".format((epoch+1),
              accuracy_epoch*100, loss_epoch))

        # save model to disk
        global_step.assign(epoch).eval()
        saver.save(sess, "./model.ckpt", global_step=global_step) 
                                # ckpt = check point

#! Testing =====================================================================
#* load testing data
print("\nLoading testing data ...")
test = pd.read_csv("./data/test.csv").values
test_x = np.array(test, dtype=np.float32)

with tf.Session() as sess:
    sess.run(init)
    print("Loading neural network model ...")
    saver.restore(sess, "model.ckpt-19") # load model: choose the one that has
                                         # the best accuracy

    y_predict = predictions.eval(feed_dict={x: test_x[1:100, :]})

    y_preditct_all = list()

    for i in np.arange(100, 8001, 100):
        y_predict = predictions.eval(feed_dict={x: test_x[i - 100:i, :]})
        test_pred = np.argmax(y_predict, axis=1)
        y_preditct_all = np.append(y_preditct_all, test_pred)

    submission = pd.DataFrame({"ImageId": range(1, 8001),
                               "Label": np.int32(y_preditct_all)})
    submission.to_csv("submission.csv", index=False)

    print("Testing is finished !")


#! Tesing 1 single image =======================================================
# import scipy as sp
# from PIL import Image

# def getTestPicArray(filename):
#     im = Image.open(filename)
#     # x_s = 28
#     # y_s = 28
#     # out = im.resize((x_s, y_s), Image.ANTIALIAS)
    
#     im_arr = np.array(im.convert('L'), dtype=np.float32)

#     # num0 = 0
#     # num255 = 0
#     # threshold = 100

#     # for x in range(x_s):
#     #     for y in range(y_s):
#     #         if im_arr[x][y] > threshold:
#     #             num255 = num255 + 1
#     #         else:
#     #             num0 = num0 + 1

#     # if(num255 > num0):
#     #     print("convert!")
#     #     for x in range(x_s):
#     #         for y in range(y_s):
#     #             im_arr[x][y] = 255 - im_arr[x][y]
#     #             if(im_arr[x][y] < threshold):
#     #                 im_arr[x][y] = 0
#                 #if(im_arr[x][y] > threshold) : im_arr[x][y] = 0
#                 #else : im_arr[x][y] = 255
#                 #if(im_arr[x][y] < threshold): im_arr[x][y] = im_arr[x][y] - im_arr[x][y] / 2

#     # out = Image.fromarray(np.uint8(im_arr))
#     # out.save(filename.split('/')[0] + '/28pix/' + filename.split('/')[1])
#     #print im_arr
#     nm = im_arr.reshape((1, 784))
#     # nm = nm.astype(np.float32)

#     # nm = np.multiply(nm, 1.0 / 255.0)
    
#     return nm

# def testMyPicture() :
#     with tf.Session() as sess:
#         sess.run(init)
#         saver.restore(sess, "model.ckpt-19")  # load model

#         testPicture = "1.jpg"
#         oneTestx = getTestPicArray(testPicture)
#         # oneTestx = test_x[1:3,:]

#         y_predict = predictions.eval(feed_dict={x: oneTestx})
#         test_pred = np.argmax(y_predict, axis=1)
#         label_predict = np.int32(y_predict)

#         print("The prediction answer is: {}".format(label_predict))


# testMyPicture()
