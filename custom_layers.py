import tensorflow as tf
from io_transform import *
import numpy as np
from math import pi
from tensorflow.keras import layers
from scipy.fftpack import fft2
import time
import os

class scale_layer(layers.Layer):

    def __init__(self, ppp, padding):
        super(scale_layer, self).__init__()
        self.scale = ppp
        self.padding = padding

    def call(self, x):
        sx = tf.map_fn(self.apply_scale, x)
        return sx

    def apply_scale(self, x):
        sx = tf.identity(x)
        paddings = tf.constant([[self.padding, self.padding,],
                                [self.padding, self.padding]])
        sx = tf.pad(sx, paddings, 'CONSTANT')
        sx_shape = tf.shape(sx)
        sx = tf.reshape(sx, [sx_shape[0], sx_shape[1], 1])
        sx = tf.tile(sx, [1, 1, self.scale])
        sx = tf.reshape(sx, [sx_shape[0], sx_shape[1]*self.scale])
        sx = tf.reshape(sx, [sx_shape[0], sx_shape[1]*self.scale, 1])
        sx = tf.tile(sx, [1, 1, self.scale])
        sx = tf.transpose(sx, [0, 2, 1])
        sx = tf.reshape(sx, [sx_shape[0]*self.scale, sx_shape[1]*self.scale])
        return sx

class grating_layer(layers.Layer):
    ''' Generates grating multiplication layer
        Spatially multiplies input with grating layer
    '''
    def __init__(self, ppp, half_period_size_1d, input_dataset_size_1d):
        super(grating_layer, self).__init__()
        self.kernel = self.grating_kernel(ppp, half_period_size_1d, input_dataset_size_1d)

    def call(self, imagex, imagey):
        return [tf.map_fn(self.apply_grating, imagex), tf.map_fn(self.apply_grating, imagey)]

    def apply_grating(self, image):
        return tf.math.multiply(image, self.kernel)

    def grating_kernel(self, ppp, half_period_size_1d, input_dataset_size_1d):
        ''' Generates binary grating of
            (ppp*input_dataset_size_1d, half_period_size_1d*input_dataset_size_1d)
        '''
        num_gratings = tf.constant([int(input_dataset_size_1d*ppp/half_period_size_1d/2)], 'int64')
        x = tf.linspace(float(0), float(2*pi), int(half_period_size_1d*2))
        unit_grating = tf.math.sin(x, 'complex128')
        grating_1d = tf.tile(unit_grating, num_gratings, 'complex128')
        [grating_2dx, grating_2dy] = tf.meshgrid(grating_1d, grating_1d)
        grating_2d = tf.math.multiply(grating_2dx, grating_2dy)
        return tf.cast(grating_2d, 'complex128')

class fsp_layer(layers.Layer):
    ''' Propagator layer with fft convolution implementation
        fft empirically determined to be faster for large kernels
        Input to fsp layer must be physical values
    '''
    def __init__(self, h, grid_size_1d, image_size_1d, prop_dist, wavelength):
        super(fsp_layer, self).__init__()
        self.grid_size_1d = tf.constant(grid_size_1d, 'float64')
        self.image_size_1d = tf.constant(image_size_1d, 'int64')
        self.prop_dist = tf.constant(prop_dist, 'complex128')
        self.wavelength = tf.constant(wavelength, 'complex128')
        self.kappa = 1/self.wavelength
        self.kernel = h

    def call(self, imagex, imagey):
        conv_x = tf.map_fn(self.apply_conv, imagex)
        conv_y = tf.map_fn(self.apply_conv, imagey)
        return [conv_x, conv_y]

    def apply_conv(self, image):
        image = tf.cast(image, 'complex128')
        conv = self.conv2dfft(self.kernel, image)
        conv = conv/tf.cast(tf.math.reduce_max(tf.abs(conv)), 'complex128')
        return conv

    def conv2dfft(self, A, B):
        # fftA = tf.signal.fft2d(A)
        fftA = A
        fftB = tf.signal.fft2d(B)
        fftAB = tf.math.multiply(fftA, fftB)
        shifted_fftAB = self.fftshift2d(fftAB)
        AB = tf.signal.ifft2d(shifted_fftAB)
        return self.ifftshift2d(AB)

    def fftshift2d(self, X):
        return tf.roll(X, shift = [tf.cast(self.image_size_1d//2, 'int64'),
                       tf.cast(self.image_size_1d//2, 'int64')], axis = [0, 1])

    def ifftshift2d(self, X):
        return tf.roll(X, shift = [tf.cast(self.image_size_1d//2, 'int64'),
                       tf.cast(self.image_size_1d//2, 'int64')], axis = [0, 1])

    # def rsfsp_kernel(self):
    #     ''' Implements Rayleigh-Sommerfeld Free Space Propagation '''
    #     rhox = tf.linspace(-self.grid_size_1d/2.0, self.grid_size_1d/2.0,
    #                        self.image_size_1d)
    #     [gridrhox, gridrhoy] = tf.meshgrid(rhox, rhox)
    #     rho2 = tf.cast(tf.math.square(gridrhox) + tf.math.square(gridrhoy), 'complex128')
    #     rho2pz2 = rho2 + tf.math.square(self.prop_dist)
    #     complex1j = tf.constant(1j, 'complex128')
    #     const_pi = tf.constant(pi, 'complex128')
    #     two = tf.constant(2, 'complex128')
    #     one = tf.constant(1, 'complex128')
    #     rsfsp = -complex1j*self.kappa*self.prop_dist/rho2pz2*tf.math.exp(complex1j*two*const_pi*self.kappa*tf.math.sqrt(rho2pz2))*(one + complex1j/(two*const_pi*self.kappa*tf.sqrt(rho2pz2)))
    #
    #     return rsfsp

class lc_layer(layers.Layer):
    ''' Implements Liquid Crystal layer
        Make 28x28 Weights
        Scale weight matrix to image size
        Update weights
    '''

    def __init__(self, input_dataset_size_1d, image_size_1d, d, wavelength, batch_size):

        ''' Parameters needed
            d = liquid crystal length
        '''
        super(lc_layer, self).__init__()
        self.input_dataset_size_1d = input_dataset_size_1d
        self.image_size_1d = image_size_1d
        self.batch_size = batch_size
        self.scale = int(image_size_1d/input_dataset_size_1d)

        self.V = tf.Variable(initial_value = np.random.rand(input_dataset_size_1d, input_dataset_size_1d)*2*pi, trainable = True)
        self.Vc = tf.constant(1.0, 'complex128')
        self.Vo = tf.constant(1.0, 'complex128')

        self.ne = tf.constant(1.2, 'complex128')
        self.no = tf.constant(1.06, 'complex128')
        self.wavelength = wavelength

        self.d = tf.constant(d, 'complex128')

    def call(self, imagex, imagey):
        scaled_V = self.scale_weights()
        [outputx, outputy] = self.lc(imagex, imagey, scaled_V)
        return [outputx, outputy]

    def scale_weights(self):
        ''' Scales weights from nxn to n*scale x n*scale
            For example, scales 28x28 to 2800x2800 if scale = 100, each 1 pixel
                is scaled to 100x100
        '''
        w = tf.identity(self.V)
        w_shape = tf.shape(w)
        w = tf.reshape(w, [w_shape[0], w_shape[1], 1])
        w = tf.tile(w, [1, 1, self.scale])
        w = tf.reshape(w, [w_shape[0], w_shape[1]*self.scale])
        w = tf.reshape(w, [w_shape[0], w_shape[1]*self.scale, 1])
        w = tf.tile(w, [1, 1, self.scale])
        w = tf.transpose(w, [0, 2, 1])
        w = tf.reshape(w, [1, w_shape[0]*self.scale, w_shape[1]*self.scale])
        w = tf.tile(w, [self.batch_size, 1, 1])
        return w

    def lc(self, imagex, imagey, scaled_V):
        ''' Apply Liquid Crystal Function '''
        pi_term = tf.constant(pi, 'complex128')
        two = tf.constant(2.0, 'complex128')
        scaled_V = tf.cast(scaled_V, 'complex128')
        one = tf.constant(1.0, 'complex128')

        # compute theta
        pi2_term = pi_term/two
        exp_term = tf.cast(tf.math.exp(-(scaled_V - self.Vc)/self.Vo), 'float64')
        atan_term = tf.cast(tf.cast(two, 'float64')*tf.atan(exp_term), 'complex128')
        theta_nonzero = pi2_term - atan_term

        scaled_V = tf.cast(scaled_V, 'float64')
        bool_V = tf.cast((scaled_V > tf.cast(self.Vc, 'float64')), 'complex128')
        scaled_V = tf.cast(scaled_V, 'complex128')
        theta = theta_nonzero*bool_V

        # compute ne_theta
        cos_term = tf.math.square(tf.math.cos(theta)/self.ne)
        sin_term = tf.math.square(tf.math.sin(theta)/self.no)
        ne_theta = tf.math.sqrt(one/(cos_term + sin_term))
        # compute beta
        pidlambda_term = pi_term*self.d/self.wavelength
        beta = pidlambda_term*(ne_theta - self.no)
        # compute phi
        phi = pidlambda_term*(ne_theta + self.no)
        #compute gamma
        gamma = tf.math.sqrt(tf.math.square(pi_term/two) + tf.math.square(beta))

        sin_gamma = tf.math.sin(gamma)
        cos_gamma = tf.math.cos(gamma)
        imag_j = tf.constant(1j, 'complex128')
        expjphi = tf.math.exp(-imag_j*phi)

        outputx = expjphi*(imagex*(pi_term/(two*gamma))*sin_gamma + imagey*(cos_gamma + imag_j*(beta/gamma)*sin_gamma))
        outputy = expjphi*(imagex*(-cos_gamma + imag_j*(beta/gamma)*sin_gamma + imagey*(pi/(two*gamma))*sin_gamma))

        return [outputx, outputy]

class eval_layer(layers.Layer):
    ''' Converts output to one hot form
        Applies softmax

        slice matrix/unpad
        pool
    '''
    def __init__(self, ppp, input_dataset_size_1d, image_size_1d, padding):
        super(eval_layer, self).__init__()
        self.input_dataset_size_1d = input_dataset_size_1d
        self.image_size_1d = image_size_1d
        self.padding = padding
        self.scale = image_size_1d/input_dataset_size_1d
        self.pooling_window_size = int(self.scale/4)
        self.scaled_padding_size = int(padding*self.scale)
        self.ppp = ppp

    def call(self, imagex):
        Ix = tf.cast(tf.math.square(tf.math.abs(imagex)), 'float64')
        r = tf.map_fn(self.eval, Ix)
        r = tf.cast(r, 'float64')
        return r

    def eval(self, Ix):
        im = self.unpad(Ix)
        ws = self.ppp*(self.input_dataset_size_1d - 2*self.padding)/4
        pooled =  tf.nn.pool(tf.cast(tf.reshape(im,
                            [1, im.shape[0], im.shape[1], 1]),'float64'),
                            window_shape = (ws, ws), pooling_type = 'AVG',
                            padding = 'SAME', strides = [ws, ws])
        pooled = tf.reshape(tf.reshape(pooled, [4, 4]), [1, 16])[0][0:10]
        pooled = tf.cast(pooled, 'float64')
        r = tf.cast(tf.nn.softmax(pooled), 'float64')
        return r

    def unpad(self, X):
        return X[self.scaled_padding_size:-self.scaled_padding_size,
                 self.scaled_padding_size:-self.scaled_padding_size]

''' Layer initialization '''
def s(imagex, ppp, padding):
    sl = scale_layer(ppp, padding)
    return sl(imagex)

def g(imagex, imagey, ppp, half_period_size_1d, input_dataset_size_1d):
    gl = grating_layer(ppp, half_period_size_1d, input_dataset_size_1d)
    return gl(imagex, imagey)

def f(imagex, imagey, h, grid_size_1d, image_size_1d, prop_dist, wavelength):
    fspl = fsp_layer(h, grid_size_1d, image_size_1d, prop_dist, wavelength)
    return fspl(imagex, imagey)

def l(imagex, imagey, input_dataset_size_1d, image_size_1d, lc_length, wavelength, batch_size):
    lcl = lc_layer(input_dataset_size_1d, image_size_1d, lc_length, wavelength, batch_size)
    return lcl(imagex, imagey)

def e(imagex, imagey, ppp, input_dataset_size_1d, image_size_1d, padding):
    el = eval_layer(ppp, input_dataset_size_1d, image_size_1d, padding)
    return el(imagex)


if __name__ == '__main__':
    source_size = 6e-6
    prop_dist = 3e-3
    wavelength = 500e-9
    k = 1/wavelength
    lc_length = 1e-4

    padding = 5
    input_dataset_size_1d = 28 + padding*2
    ppp = 40
    norm_std = 0.3
    output_value = 1
    half_period_size_1d = 10

    image_size_1d = input_dataset_size_1d*ppp
    grid_size_1d = input_dataset_size_1d*source_size

    batch_size = 32

    print('Layers Built')

    (x_train, x_test), (y_train, y_test) = load_mnist_data()
    imagex = tf.constant(x_train[0:batch_size], 'complex128')

    # generate free space propagator
    x = np.linspace(-grid_size_1d/2.0, grid_size_1d/2.0, image_size_1d)
    [rhox, rhoy] = np.meshgrid(x, x)
    rho2 = np.sqrt(rhox**2 + rhoy**2)
    z = prop_dist
    H = -1j*k*z/(rho2+z**2)*np.exp(1j*2*pi*k*np.sqrt(rho2 + z**2))*(1+ 1j/2/pi/k/np.sqrt(rho2+z**2))
    h = fft2(H)
    h = tf.constant(h, 'complex128')


    with tf.Session() as sess:
        start = time.time()
        imagex = s(imagex, ppp, padding)
        imagey = tf.constant(np.zeros((batch_size, image_size_1d, image_size_1d)), 'complex128') + tf.constant(1e-10, 'complex128')

        [imagex, imagey] = g(imagex, imagey, ppp, half_period_size_1d, input_dataset_size_1d)
        [imagex, imagey] = f(imagex, imagey, h, grid_size_1d, image_size_1d, prop_dist, wavelength)
        [imagex, imagey] = l(imagex, imagey, input_dataset_size_1d, image_size_1d, lc_length, wavelength, batch_size)

        # [imagex, imagey] = g(imagex, imagey, ppp, half_period_size_1d, input_dataset_size_1d)
        # [imagex, imagey] = f(imagex, imagey, h, grid_size_1d, image_size_1d, prop_dist, wavelength)
        # [imagex, imagey] = l(imagex, imagey, input_dataset_size_1d, image_size_1d, lc_length, wavelength, batch_size)
        #
        # [imagex, imagey] = g(imagex, imagey, ppp, half_period_size_1d, input_dataset_size_1d)
        # [imagex, imagey] = f(imagex, imagey, h, grid_size_1d, image_size_1d, prop_dist, wavelength)
        # [imagex, imagey] = l(imagex, imagey, input_dataset_size_1d, image_size_1d, lc_length, wavelength, batch_size)
        #
        # [imagex, imagey] = g(imagex, imagey, ppp, half_period_size_1d, input_dataset_size_1d)
        # [imagex, imagey] = f(imagex, imagey, h, grid_size_1d, image_size_1d, prop_dist, wavelength)
        # [imagex, imagey] = l(imagex, imagey, input_dataset_size_1d, image_size_1d, lc_length, wavelength, batch_size)
        #
        # [imagex, imagey] = g(imagex, imagey, ppp, half_period_size_1d, input_dataset_size_1d)
        # [imagex, imagey] = f(imagex, imagey, h, grid_size_1d, image_size_1d, prop_dist, wavelength)
        # [imagex, imagey] = l(imagex, imagey, input_dataset_size_1d, image_size_1d, lc_length, wavelength, batch_size)

        # out = e(imagex, imagey, ppp, input_dataset_size_1d, image_size_1d, padding)

        sess.run(tf.global_variables_initializer())
        plt.figure()
        x = imagex.eval()[0, :, :]
        plt.title('Field Intensity x-polarized')
        plt.imshow(np.absolute(x)**2)
        plt.colorbar()
        plt.figure()
        y = imagey.eval()[0, :, :]
        plt.title('Field Intensity y-polarized')
        plt.imshow(np.absolute(y)**2)
        plt.colorbar()
        plt.show()

        end = time.time()
        print(end - start)
