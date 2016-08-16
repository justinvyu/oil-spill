
import tensorflow as tf

import numpy as np
from numpy import random

import math
from enum import Enum

class Trainer(object):
    """Wrapper class around Tensorflow softmax regression

    Attributes:
        x: input placholder that holds an input matrix of size [n_samples, n_features]
        y_: output placeholder that holds one-hot vector labels
        y: output of softmax regression (predicted one-hot vector label)
        W: weight matrix of size [n_features, n_classes]
        b: bias matrix

        +n_features: number of features per sample
        +n_classes: number of classes / categories
        +dropout: probability to keep units
    """

    n_features = 25
    n_classes = 2
    dropout = 0.75

    def __init__(self, training_input, training_labels, testing_input, testing_labels,
                learning_rate=4e-5, training_iters=2000) :

        self.training_input = training_input
        self.training_labels = training_labels
        self.testing_input = testing_input
        self.testing_labels = testing_labels

        self.sess = tf.InteractiveSession()

        # Hyper-parameters
        self.training_iters = training_iters
        self.learning_rate = learning_rate

        # Model - TODO: Cleanup

        self.x = tf.placeholder(tf.float32, [None, Trainer.n_features])
        # self.x = tf.Print(self.x, [self.x], "Input: ")

        self.W = tf.Variable(tf.zeros([Trainer.n_features, Trainer.n_classes]))
        # self.W = tf.Print(self.W, [self.W], "Weights: ")

        self.y_ = tf.placeholder(tf.float32, [None, Trainer.n_classes]) # labels

        self.b = tf.Variable(tf.zeros([Trainer.n_classes]))
        # self.b = tf.Print(self.b, [self.b], "Bias: ")

        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        # self.y = tf.Print(self.y, [self.y], "Output: ")

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.sess.run(tf.initialize_all_variables())
        # self.recalibrate(Trainer.n_features)

    def recalibrate(self, n_features) :
        self.x = tf.placeholder(tf.float32, [None, n_features])
        # self.x = tf.Print(self.x, [self.x], "Input: ")

        self.W = tf.Variable(tf.zeros([n_features, Trainer.n_classes]))
        # self.W = tf.Print(self.W, [self.W], "Weights: ")

        self.y_ = tf.placeholder(tf.float32, [None, Trainer.n_classes]) # labels

        self.b = tf.Variable(tf.zeros([Trainer.n_classes]))
        # self.b = tf.Print(self.b, [self.b], "Bias: ")

        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        # self.y = tf.Print(self.y, [self.y], "Output: ")

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def filter_cols(self, _input, feature_filter) :
        temp_input = _input
        n_features = 0
        n_samples = _input.shape[0]

        temp_input = np.array([]).reshape(n_samples, 0)
        for i, index in enumerate(feature_filter):
            if index != 0:
                col = _input[:, i].reshape((n_samples, 1))
                temp_input = np.hstack((temp_input, col))

                n_features += 1

        temp_input = np.reshape(temp_input, (-1, sum(feature_filter)))

        if n_features <= 0:
            return None

        return temp_input, n_features

    def train(self, feature_filter=[]) :
        filtered = self.training_input
        if len(feature_filter) > 0:
            filtered, n_features = self.filter_cols(self.training_input, feature_filter)
            if filtered is None:
                return None
            self.recalibrate(n_features)
            self.sess.run(tf.initialize_all_variables())

        for i in range(self.training_iters):
            self.sess.run(self.train_step, feed_dict={ self.x: filtered,
                                                       self.y_: self.training_labels })

    def percent_accuracy(self, feature_filter=[]) :
        filtered = self.testing_input
        if len(feature_filter) > 0:
            filtered, n_features = self.filter_cols(self.testing_input, feature_filter)
            if filtered is None:
                return None

        return self.sess.run([self.accuracy, self.correct_prediction], feed_dict={ self.x: filtered,
                                                        self.y_: self.testing_labels })

    # def weight_variable(shape) :
    #     initial = tf.truncated.normal(shape, stddev=0.1)
    #     return tf.Variable(initial)
    #
    # def bias_variable(shape) :
    #     initial = tf.constant(0.1, shape=shape)
    #     return tf.Variable(initial)
