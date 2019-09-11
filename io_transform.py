import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from PIL import Image
import pickle

def scale_image(x, scale):
    x_shape = x.shape
    x = x.reshape([x_shape[0], x_shape[1], 1])
    x = np.tile(x, [1, 1, scale])
    x = x.reshape([x_shape[0], x_shape[1]*scale])
    x = x.reshape([x_shape[0], x_shape[1]*scale, 1])
    x = np.tile(x, [1, 1, scale])
    x = np.transpose(x, [0, 2, 1])
    x = x.reshape([x_shape[0]*scale, x_shape[1]*scale])
    return x

def image_to_beam(image, ppp, norm_std, padding):
    ''' Option: Generates array of Gaussian Beams
                Amplitude of Gaussian beam scaled to image pixel value
        Matrix multiply takes too much time. Assume uniform beam
    '''
    output_image_size_1d = ppp*image.shape[0]
    # beam_1d = [0]*int(ppp/4) + [1]*int(ppp/2) + [0]*int(ppp/4)
    # [xbeam, ybeam] = np.sqrt(np.meshgrid(beam_1d, beam_1d))
    # beam_array = np.tile(xbeam*ybeam, (image.shape[0], image.shape[0]))
    simage = scale_image(image, ppp)
    # image_beam_array = np.multiply(beam_array, simage)
    # image_beam_array = simage
    # if padding > 0:
    #     image_beam_array = np.pad(image_beam_array.astype('float64'), padding*ppp, 'constant')
    return simage

# def output_transform(value, input_size_1d, ppp, output_value, padding):
#     ''' Generates desired output
#         Makes grid of 16 values (only 10 used)
#         Sclaed grid to required output size
#     '''
#     output_size_1d = input_size_1d*ppp
#     grid = np.zeros((1, 16))
#     grid[0, value] = output_value
#     rgrid = grid.reshape((4, 4))
#     sgrid = scale_image(rgrid, int(output_size_1d/4))
#     if padding > 0:
#         sgrid = np.pad(sgrid.astype('float64'), padding*ppp, 'constant')
#     return sgrid

def prepare_mnist_data(ppp, norm_std, output_value):
    ''' Loads MNIST data
        Applies image_to_beams to x arrays
        Applies output_transform to y arrays
        returns transformed x, y, train, test, validation arrays
    '''
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    x_train = x_train.astype('float64')
    y_train = y_train.astype('float64')
    x_test = x_test.astype('float64')
    y_test = y_test.astype('float64')
    print('Data Loaded')
    x_train_beams = np.zeros((x_train.shape[0],
                        x_train.shape[1]*ppp, x_train.shape[2]*ppp))
    for i in range(x_train.shape[0]):
        x_train_beams[i, :, :] = image_to_beam(x_train[i, :, :], ppp, norm_std, padding)
        print('Completed ' + str(i) + '/' + str(x_train.shape[0]) + ' Train Set')
    x_test_beams = np.zeros((x_test.shape[0],
                       x_test.shape[1]*ppp, x_test.shape[2]*ppp))
    for i in range(x_test.shape[0]):
        x_test_beams[i, :, :] = image_to_beam(x_test[i, :, :], ppp, norm_std, padding)
        print('Completed ' + str(i) + '/' + str(x_test.shape[0]) + ' Test Set')
    print('mnist data beams loaded')

    # pickle data
    data = (x_train_beams, x_test_beams), (y_train, y_test)
    with open('mnist_beam.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return (x_train_beams, x_test_beams), (y_train, y_test)

def prepare_one_mnist(ppp, norm_std, output_value, padding):
    ''' Prepares one MNIST handwritten image and converts into beams for testing
    '''
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print('Loaded Data')
    print('\tx_train shape = ' + str(x_train.shape))
    print('\ty_train shape = ' + str(y_train.shape))
    print('\tx_test shape = ' + str(x_test.shape))
    print('\ty_test shape = ' + str(y_test.shape))
    image_beam = image_to_beam(x_train[0], ppp, norm_std, padding)
    return image_beam/np.max(image_beam)

def pickle_mnist():
    from tensorflow.keras.datasets.mnist import load_data
    (x_train, y_train), (x_test, y_test) = load_data()
    data = (x_train, y_train), (x_test, y_test)
    with open('mnist.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_mnist_data():
    with open('mnist.pickle', 'rb') as f:
        print('Loading Pickled Data...')
        return pickle.load(f)

if __name__ == '__main__':
    ppp = 40
    output_value = 1
    norm_std = 0.3
    padding = 5

    # pickle_mnist()
    # (x_train, y_train), (x_test, y_test) = load_mnist_data()
    # print(x_train.shape, x_test.shape)

    (x_train_beams, x_test_beams), (y_train, y_test) = prepare_mnist_data(ppp, norm_std, output_value)

    # image_beam = prepare_one_mnist(ppp, norm_std, output_value, padding)
    # output_beam = output_transform(0, x_train[0].shape[0], ppp, output_value, padding)
    # plt.figure()
    # plt.imshow(image_beam)
    # plt.show()
    # plt.imshow(output_beam)
    # plt.show()
    # [x_train, y_train, x_test, y_test] = prepare_mnist_data(ppp, norm_std, output_value)
