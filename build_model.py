from custom_layers import *

class d2nn_model(object):
    def __init__(self):
        #super(YourModel, self).__init__()
        self.num_epoch = 15
        self.batch_size = 10
        self.log_step = 100
        self.lr = 1e-3

        # Define model parameters
        self.source_size = 6e-6
        self.prop_dist = 3e-3
        self.wavelength = 500e-9
        self.k = 1.0/self.wavelength
        self.lc_length = 1e-4
        self.padding = 5
        self.input_dataset_size_1d = 28 + self.padding*2
        self.ppp = 40
        self.norm_std = 0.3
        self.output_value = 1
        self.half_period_size_1d = 10
        self.image_size_1d = self.input_dataset_size_1d*self.ppp
        self.grid_size_1d = self.input_dataset_size_1d*self.source_size

        # generate free space propagator
        x = np.linspace(-self.grid_size_1d/2.0, self.grid_size_1d/2.0, self.image_size_1d)
        [rhox, rhoy] = np.meshgrid(x, x)
        rho2 = np.sqrt(rhox**2 + rhoy**2)
        z = self.prop_dist
        k = self.k
        H = -1j*k*z/(rho2+z**2)*np.exp(1j*2*pi*k*np.sqrt(rho2 + z**2))*(1+ 1j/2/pi/k/np.sqrt(rho2+z**2))
        h = tf.constant(fft2(H), 'complex128')
        self.h = h

        self._build_model()

    def _model(self):
        print('-' * 5 + '  Your model  ' + '-' * 5)
        print('-' * 5 + '  Sample model  ' + '-' * 5)
        print('input layer: ' + str(self.imagex.get_shape()))
        with tf.variable_scope('scale1'):
            self.imagex0 = s(self.imagex, self.ppp, self.padding)
            self.imagey0 = s(self.imagey, self.ppp, self.padding)
        print('imagex shape:\t', self.imagex)
        print('imagey shape:\t', self.imagex)
        with tf.variable_scope('unit1'):
            [self.imagex1, self.imagey1] = g(self.imagex0, self.imagey0, self.ppp, self.half_period_size_1d, self.input_dataset_size_1d)
            [self.imagex1, self.imagey1] = f(self.imagex1, self.imagey1, self.h, self.grid_size_1d, self.image_size_1d, self.prop_dist, self.wavelength)
            [self.imagex1, self.imagey1] = l(self.imagex1, self.imagey1, self.input_dataset_size_1d, self.image_size_1d, self.lc_length, self.wavelength, self.batch_size)
            print('unit1 layer: ' + str(self.imagex1.get_shape()))
        with tf.variable_scope('unit2'):
            [self.imagex2, self.imagey2] = g(self.imagex1, self.imagey1, self.ppp, self.half_period_size_1d, self.input_dataset_size_1d)
            [self.imagex2, self.imagey2] = f(self.imagex2, self.imagey2, self.h, self.grid_size_1d, self.image_size_1d, self.prop_dist, self.wavelength)
            [self.imagex2, self.imagey2] = l(self.imagex2, self.imagey2, self.input_dataset_size_1d, self.image_size_1d, self.lc_length, self.wavelength, self.batch_size)
            print('unit2 layer: ' + str(self.imagex2.get_shape()))
        with tf.variable_scope('unit3'):
            [self.imagex3, self.imagey3] = g(self.imagex2, self.imagey2, self.ppp, self.half_period_size_1d, self.input_dataset_size_1d)
            [self.imagex3, self.imagey3] = f(self.imagex3, self.imagey3, self.h, self.grid_size_1d, self.image_size_1d, self.prop_dist, self.wavelength)
            [self.imagex3, self.imagey3] = l(self.imagex3, self.imagey3, self.input_dataset_size_1d, self.image_size_1d, self.lc_length, self.wavelength, self.batch_size)
            print('unit3 layer: ' + str(self.imagex.get_shape()))
        with tf.variable_scope('unit4'):
            [self.imagex4, self.imagey4] = g(self.imagex3, self.imagey3, self.ppp, self.half_period_size_1d, self.input_dataset_size_1d)
            [self.imagex4, self.imagey4] = f(self.imagex4, self.imagey4, self.h, self.grid_size_1d, self.image_size_1d, self.prop_dist, self.wavelength)
            [self.imagex4, self.imagey4] = l(self.imagex4, self.imagey4, self.input_dataset_size_1d, self.image_size_1d, self.lc_length, self.wavelength, self.batch_size)
            print('unit4 layer: ' + str(self.imagex4.get_shape()))
        with tf.variable_scope('unit5'):
            [self.imagex5, self.imagey5] = g(self.imagex4, self.imagey4, self.ppp, self.half_period_size_1d, self.input_dataset_size_1d)
            [self.imagex5, self.imagey5] = f(self.imagex5, self.imagey5, self.h, self.grid_size_1d, self.image_size_1d, self.prop_dist, self.wavelength)
            [self.imagex5, self.imagey5] = l(self.imagex5, self.imagey5, self.input_dataset_size_1d, self.image_size_1d, self.lc_length, self.wavelength, self.batch_size)
            print('unit5 layer: ' + str(self.imagex5.get_shape()))
        with tf.variable_scope('eval1'):
            self.eval1 = e(self.imagex5, self.imagey5, self.ppp, self.input_dataset_size_1d, self.image_size_1d, self.padding)
            print('self eval 1:\t : ', self.eval1)
            print('eval1 layer: ' + str(self.eval1.get_shape()))
        return self.eval1

    def _input_ops(self):
        # Placeholders
        self.imagex = tf.placeholder(tf.complex128, [self.batch_size, None, None])
        self.imagey = tf.placeholder(tf.complex128, [self.batch_size, None, None])
        self.Y = tf.placeholder(tf.int64, [self.batch_size])
        self.is_train = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float64)

    def _build_optimizer(self):
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss_op)

    def _loss(self, labels, logits):
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.Y,10), logits=self.eval1))

    def _build_model(self):
        # Define input variables
        self._input_ops()
        # Convert Y to one-hot vector
        labels = tf.one_hot(self.Y, 10)
        # Build a model and get logits
        logits = self._model()
        # Compute loss
        self._loss(labels, logits)
        # Build optimizer
        self._build_optimizer()
        # Compute accuracy
        predict = tf.argmax(logits, 1)
        correct = tf.equal(predict, self.Y)
        self.accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float64))

    def train(self, sess, XX_train, XY_train, Y_train, X_val, Y_val):
        sess.run(tf.global_variables_initializer())
        step = 0
        losses = []
        accuracies = []
        print('-' * 5 + '  Start training  ' + '-' * 5)
        for epoch in range(self.num_epoch):
            XX_train, Y_train = shuffle(XX_train, Y_train)
            print('train for epoch %d' % epoch)
            for i in range(XX_train.shape[0] // self.batch_size):
                XX_ = XX_train[i * self.batch_size:(i + 1) * self.batch_size][:]
                XY_ = XY_train[i * self.batch_size:(i + 1) * self.batch_size][:]
                Y_ = Y_train[i * self.batch_size:(i + 1) * self.batch_size]
                feed_dict = {self.imagex: XX_,
                             self.imagey: XY_,
                             self.Y: Y_,
                             self.is_train: True}
                fetches = [self.train_op, self.loss_op, self.accuracy_op]

                _, loss, accuracy = sess.run(fetches, feed_dict=feed_dict)
                losses.append(loss)
                accuracies.append(accuracy)

                if step % self.log_step == 0:
                    print('iteration (%d): loss = %.3f, accuracy = %.3f' %
                        (step, loss, accuracy))
                step += 1
                if step % 3000 == 0:
                    self.lr = self.lr*0.9

            # Plot training curves
            plt.subplot(2, 1, 1)
            plt.plot(losses)
            plt.grid(True)
            plt.xlabel('Epoch {}'.format(epoch+1))
            plt.ylabel('Training Loss')
            plt.show()
            # Graph 2. X: epoch, Y: training accuracy
            plt.subplot(2, 1, 2)
            plt.plot(accuracies)
            plt.grid(True)
            plt.xlabel('Epoch {}'.format(epoch+1))
            plt.ylabel('Training Accuracy')
            plt.show()

            # Print validation results
            print('validation for epoch %d' % epoch)
            val_accuracy = self.evaluate(sess, X_val, Y_val)
            print('-  epoch %d: validation accuracy = %.3f' % (epoch, val_accuracy))

    def evaluate(self, sess, X_eval, Y_eval):
        eval_accuracy = 0.0
        eval_iter = 0
        for i in range(X_eval.shape[0] // self.batch_size):
            X_ = X_eval[i * self.batch_size:(i + 1) * self.batch_size][:]
            Y_ = Y_eval[i * self.batch_size:(i + 1) * self.batch_size]

            feed_dict = {self.X: X_,
                            self.Y: Y_,
                            self.is_train: False}

            accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)
            eval_accuracy += accuracy
            eval_iter += 1
        return eval_accuracy / eval_iter

def shuffle(X, Y):
    train_indices = np.arange(X.shape[0])
    np.random.shuffle(train_indices)
    X_result = X[train_indices,:, :]
    Y_result = Y[train_indices]
    return X_result, Y_result

if __name__ == '__main__':
    # Clear old computation graphs
    tf.reset_default_graph()

    with tf.Session() as sess:

        (x_train, y_train), (x_test, y_test) = load_mnist_data()

        x_val = x_train[-10000:, :, :].astype('float64')
        y_val = y_train[-10000:].astype('float64')
        xx_train = x_train[:-10000, :, :].astype('float64')
        xy_train = np.zeros(xx_train.shape).astype('float64') + 1e-15
        y_train = y_train[:-10000].astype('float64')
        x_test = x_test.astype('float64')
        y_test = y_test.astype('float64')

        with tf.device('/cpu:0'):
            model = d2nn_model()
            model.train(sess, xx_train, xy_train, y_train, x_val, y_val)
            accuracy = model.evaluate(sess, X_test, Y_test)
            print('***** test accuracy: %.3f' % accuracy)
            saver = tf.train.Saver()
            model_path = saver.save(sess, "d2nn_model_mnist.ckpt")
            print("Model saved in %s" % model_path)
