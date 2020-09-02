#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import scipy.io as sio
for sub in range(1, 55):

    tf.reset_default_graph()
    tf.set_random_seed(1230)

    mat_data = sio.loadmat('WHERE/IS/DATA/feature_representation_sess01_subj%d.mat' % (sub))

    trX = mat_data['train_data']
    trY = mat_data['train_labels']
    teX = mat_data['test_data']
    teY = mat_data['test_labels']

    def model(x, w, w2, w3, w_o, p_keep_conv, p_keep_hidden):
        l1a = tf.nn.relu(tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='VALID'))
        l1a = tf.nn.bias_add(l1a, b1)
        l1a = tf.nn.dropout(l1a, p_keep_conv)

        l2a = tf.nn.relu(tf.nn.conv3d(l1a, w2, strides=[1, 1, 1, 1, 1], padding='VALID'))
        l2a = tf.nn.bias_add(l2a, b2)
        l2a = tf.nn.dropout(l2a, p_keep_conv)

        l2 = tf.reshape(l2a, [-1, w3.get_shape().as_list()[0]])
        l3 = tf.nn.relu(tf.matmul(l2, w3))
        l3 = tf.nn.dropout(l3, p_keep_hidden)

        pyx = tf.matmul(l3, w_o)
        return pyx

    trX = trX.reshape(-1, 20, 20, 25, 1) 
    teX = teX.reshape(-1, 20, 20, 25, 1) 

    x = tf.placeholder("float", [None, 20, 20, 25, 1]) 
    y_ = tf.placeholder("float", [None, 2])
    
    kernel1=3
    kernel2=3
    feature_map1=50
    feature_map2=100

    w = tf.get_variable("w", shape=[kernel1, kernel1, kernel2, 1, feature_map1],
                        initializer=tf.truncated_normal_initializer(stddev=0.01))
    w2 = tf.get_variable("w2", shape=[21-kernel1, 21-kernel1, 26-kernel2, feature_map1, feature_map2],
                         initializer=tf.truncated_normal_initializer(stddev=0.01))
    w3 = tf.get_variable("w3", shape=[feature_map2, feature_map2], initializer=tf.truncated_normal_initializer(stddev=0.01))
    w_o = tf.get_variable("w_o", shape=[feature_map2, 2], initializer=tf.truncated_normal_initializer(stddev=0.01))

    b1 = tf.get_variable("b1", shape=[feature_map1], initializer=tf.constant_initializer(0.0))
    b2 = tf.get_variable("b2", shape=[feature_map2], initializer=tf.constant_initializer(0.0))

    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")
    y = model(x, w, w2, w3, w_o, p_keep_conv, p_keep_hidden)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_op = tf.train.AdamOptimizer(0.0001).minimize(cost)
    predict_op = tf.argmax(y, 1)

    itr=500
    acc = [0] * itr
    tracc = [0] * itr
    # Launch the graph in a session
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(itr):
            sess.run(train_op, feed_dict={x: trX, y_: trY, p_keep_conv: 0.8, p_keep_hidden: 0.8})
            #print(100 * np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={x: teX, y_: teY, p_keep_conv: 1.0, p_keep_hidden: 1.0})))
            acc[i] = 100 * np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={x: teX, y_: teY, p_keep_conv: 1.0, p_keep_hidden: 1.0}))
            tracc[i] = 100 * sess.run(accuracy, feed_dict={x: trX, y_: trY, p_keep_conv: 0.8, p_keep_hidden: 0.8})
        sio.savemat('result_feature_representation_sess01_subj%d.mat' % (sub), {"tracc": tracc, "acc": acc})

