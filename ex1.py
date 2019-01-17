
# TensorFlow execices
# Hao BAI
# ref: https://www.jianshu.com/p/2ea7a0632239

import tensorflow as tf

ex = 3

#* Product of matrix ===========================================================
if ex == 1:
    cst1 = tf.constant([ [2,3] ]) # a constant of 1 row 2 columns
    cst2 = tf.constant([ [2], [3] ]) # a constant of 2 rows 1 columns
    print("[Constant matrix] cst1 = {} \n cst2 = {}".format(cst1, cst2))

    product = tf.matmul(cst1, cst2) # a matrix multiply operator
    print("[Operator] product = {}".format(product))

    sess = tf.Session() # define a session
    result = sess.run(product) # run product calculation
    print("result = {}".format(result))
    sess.close() # close a session


#* Use of for-loop =============================================================
if ex == 2:
    var = tf.Variable(0, name="Count") # a variable named "count" was given an
                                       # initial value 0, but the initialisation
                                       # hasn't been executed yet
    print("[Variable] var = {}".format(var))

    somme = tf.add(var, 1) # somme = var + 1
    print("[Operator] somme = {}".format(somme))
    assignment = tf.assign(var, somme) # var = somme: assign value of somme to var
    print("[Operator] assignment = {}".format(assignment))

    print("Open a session ...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # initiate variables
        print("initialisation: sess.run(var) = {}".format(sess.run(var)))
        for i in range(5):
            sess.run(assignment)
            print("i = {} : sess.run(var) = {}".format(i, sess.run(var)))


#* feed/fetch ==================================================================
if ex == 3:
    hole1 = tf.placeholder(tf.float64, name="Holder1")
    hole2 = tf.placeholder(tf.float64, name="Holder2")
    print("[Empty Variable] hole1 = {} \n hole2 = {}".format(hole1, hole2))

    product = tf.multiply(hole1, hole2) # a multiply operator
    print("[Operator] product = {}".format(product))

    print("Open a session ...")
    with tf.Session() as sess:
        result = sess.run(product, feed_dict={hole1:33, hole2:3})
        print("result = {}".format(result))
