import numpy as np

import tensorflow as tf

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu']
allowed_noises = [None, 'gaussian', 'mask']
allowed_losses = ['rmse', 'cross-entropy']

def get_batch(X, X_, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a], X_[a]

class SdA:
    """A deep autoencoder with denoising capability"""

    def assertions(self):
        global allowed_activations, allowed_noises, allowed_losses
        assert self.loss in allowed_losses, 'Incorrect loss given'
        assert 'list' in str(type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.epoch) == len(self.dims), "No. of epochs must equal to no. of hidden layers"
        assert len(self.activations) == len(self.dims), "No. of activations must equal to no. of hidden layers"
        assert all(True if x > 0 else False for x in self.epoch), "No. of epoch must be atleast 1"
        assert set(self.activations + allowed_activations) == set(allowed_activations), "Incorrect activation given."
        assert self.noise in allowed_noises, "Incorrect noise given"

    def __init__(self, dims, activations, epoch=1000, noise=None, loss='rmse', lr=0.001, batch_size=100, print_step=50):
        self.print_step = print_step
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.epoch = epoch
        self.dims = dims
        self.assertions()
        self.depth = len(dims)
        self.weights, self.biases = [], []

    def add_noise(self, x):
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if 'mask' in self.noise:
            frac = float(self.noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), round(frac * len(i)), replace=False)
                i[n] = 0
            return temp
        if self.noise == 'sp':
            pass

    def fit(self, x):
        for i in range(self.depth):
            print('Layer {0}'.format(i + 1))
            if self.noise is None:
                x = self.run(data_x=x, activation=self.activations[i], data_x_=x,
                             hidden_dim=self.dims[i], epoch=self.epoch[i], loss=self.loss,
                             batch_size=self.batch_size, lr=self.lr, print_step=self.print_step)
            else:
                x = self.run(data_x=self.add_noise(x), activation=self.activations[i], data_x_=x,
                             hidden_dim=self.dims[i],
                             epoch=self.epoch[i], loss=self.loss, batch_size=self.batch_size, lr=self.lr,
                             print_step=self.print_step)
        print(x)

    def transform(self, data):
        sess = tf.Session()
        x = tf.constant(data, dtype=tf.float32)
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.activate(layer, a)
        return x.eval(session=sess)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def run(self, data_x, data_x_, hidden_dim, activation, loss, lr, print_step, epoch, batch_size=100):
        input_dim = len(data_x[0])
        sess = tf.Session()
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x_')
        encode = {'weights': tf.Variable(tf.truncated_normal([input_dim, hidden_dim], dtype=tf.float32)),
                  'biases': tf.Variable(tf.truncated_normal([hidden_dim], dtype=tf.float32))}
        decode = {'biases': tf.Variable(tf.truncated_normal([input_dim], dtype=tf.float32)),
                  'weights': tf.transpose(encode['weights'])}

        encoded = self.activate(tf.matmul(x, encode['weights']) + encode['biases'], activation)
        decoded = tf.matmul(encoded, decode['weights']) + decode['biases']

        # reconstruction loss
        if loss == 'rmse':
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(x_, decoded))))
        elif loss == 'cross-entropy':
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(decoded, x_))
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        sess.run(tf.initialize_all_variables())
        for i in range(epoch):
            b_x, b_x_ = get_batch(data_x, data_x_, batch_size)
            sess.run(train_op, feed_dict={x: b_x, x_: b_x_})
            if (i + 1) % print_step == 0:
                l = sess.run(loss, feed_dict={x: data_x, x_: data_x_})
                print('epoch {0}: global loss = {1}'.format(i, l))
        # debug
        # print('Decoded', sess.run(decoded, feed_dict={x: self.data_x_})[0])
        self.weights.append(sess.run(encode['weights']))
        self.biases.append(sess.run(encode['biases']))
        return sess.run(encoded, feed_dict={x: data_x_})

    def activate(self, linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')
