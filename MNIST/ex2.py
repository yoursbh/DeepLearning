
# TensorFlow execices: MNIST
# Hao BAI
# ref: https://www.jianshu.com/p/696bde1641d8

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


#! Parametrage =================================================================
# initialisation
init = tf.global_variables_initializer()
# calculate accuracy of prediction
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(predictions,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# saving model
global_step = tf.Variable(0, name='Global_Step', trainable=False)
saver = tf.train.Saver()


#! Train model =================================================================
# with tf.Session() as sess:
#     sess.run(init) # initialize
#     saver.restore(sess, "model.ckpt-BEST")  # load model if necessary
#     # if use TensorFlow implemented one-hot encoding, we need to convert its
#     # type to numpy.array
#     if one_hot_mode == 2:
#         train_labels = train_labels.eval()
#         valid_labels = valid_labels.eval()
    
#     # run mini batch SGD
#     for epoch in range(1,21,1): # run for 20 loops
#         #* at the end of each epoch, it begins backward propagation (means 
#         #* changing the weights)
#         for batch in range(n_batch):
#             #* every batch means a forward propagation is finished (all samples 
#             #* have been used to train the model)
#             # get data batch by batch
#             batch_x = train_images[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE]
#             batch_y = train_labels[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE]
#             # train model
#             sess.run(train_step, feed_dict={x:batch_x, y:batch_y})
    
#         # show accuracy for every epoch
#         loss_epoch = sess.run(loss, feed_dict={x: valid_images,
#                                                y: valid_labels})
#         accuracy_epoch = sess.run(accuracy, feed_dict={x:valid_images,
#                                                        y:valid_labels})
#         print("Epoch {}: accuracy is {:.2f}%, loss is {}".format(epoch,
#               accuracy_epoch*100, loss_epoch))

#         # save model to disk
#         global_step.assign(epoch).eval()
#         saver.save(sess, "./model.ckpt", global_step=global_step) 
#                                 # ckpt = check point

#! Testing =====================================================================
# print("\nLoad MNIST testing data ...")
# test = pd.read_csv("./data/test.csv").values
# test_x = np.array(test, dtype=np.float32)

# with tf.Session() as sess:
#     sess.run(init)
#     print("Load neural network model ...")
#     saver.restore(sess, "model.ckpt-BEST") # load model: choose the one that has
#                                          # the best accuracy

#     y_predict = predictions.eval(feed_dict={x: test_x[1:100, :]})

#     y_preditct_all = list()

#     for i in np.arange(100, 8001, 100):
#         y_predict = predictions.eval(feed_dict={x: test_x[i - 100:i, :]})
#         test_pred = np.argmax(y_predict, axis=1)
#         y_preditct_all = np.append(y_preditct_all, test_pred)

#     submission = pd.DataFrame({"ImageId": range(1, 8001),
#                                "Label": np.int32(y_preditct_all)})
#     submission.to_csv("submission.csv", index=False)

#     print("Testing is finished !")


#! Tesing 1 single image =======================================================
def get_test_picture(filename):
    image_origin = Image.open(filename)
    width = 28  # width of destination image
    height = 28  # height of destination image
    image_resized = image_origin.resize((width, height), Image.ANTIALIAS)
    im_arr = np.array(image_resized.convert('L'), dtype=np.float32)
    nm = im_arr.reshape((1, width*height))
    return nm

def recognize_digit():
    print("\nRecognizing ...")
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "model.ckpt-BEST")  # load best model
        for i in [1, 4, 7, 8]:
            # i = "Untitled"
            testPicture = "{}.jpg".format(i)  # filename of handwritten picture
            oneTestx = get_test_picture(testPicture)  # load picture
            y_predict = predictions.eval(feed_dict={x: oneTestx}) # predict the
                                                                  # digit
            test_pred = np.argmax(y_predict, axis=1) # get the one that has the
                                                     # largest probability
            label_predict = np.int32(test_pred)
            print("|- The prediction of picture {}.jpg is: {}".format(i,
                  label_predict))

recognize_digit()
