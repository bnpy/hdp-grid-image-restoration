import numpy as np
from DPGridModel import DPGridModel
from HDPGridModel import HDPGridModel
from util import imread, imwrite


def demo_eDP():
    I = imread('images/barbara.png')
    sigma = 25.0 / 255.0
    PRNG = np.random.RandomState(0)
    y = I + sigma * PRNG.randn(I.shape[0], I.shape[1])
    gridModel = DPGridModel('models/DP')
    x, PSNR = gridModel.denoise(y, sigma, I)
    imwrite(x, 'eDP_results.png')


def demo_HDP():
    I = imread('images/3096.jpg')
    sigma = 25.0 / 255.0
    PRNG = np.random.RandomState(0)
    y = I + sigma * PRNG.randn(I.shape[0], I.shape[1])
    gridModel = HDPGridModel('models/HDP')
    x, PSNR = gridModel.denoise(y, sigma, I)
    imwrite(x, 'HDP_results.png')


if __name__ == '__main__':
    demo_eDP()
    demo_HDP()