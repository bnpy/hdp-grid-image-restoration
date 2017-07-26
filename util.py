import numpy as np
from PIL import Image


def imread(fileName):
    '''
        Read in an image and return its gray-scale pixel values (normalized)
    '''
    im = Image.open(fileName)
    W, H = im.size
    try:
        im = im.convert('L')
    except:
        raise TypeError('Unable to convert image %s from %s to gray!' % (fileName, im.mode))
    result = np.zeros((H, W), dtype=np.float64)
    imPix = im.load()
    for i in xrange(W):
        for j in xrange(H):
            result[j, i] = imPix[i, j]
    return result / 255.0


def imwrite(I, fileName, fmt='PNG'):
    '''
        Write an image I (normalized) to file fileName in format fmt
    '''
    im = Image.fromarray(np.uint8(np.round(I * 255)))
    im.save(fileName, fmt)


def im2col(I, patchSize, stride=1):
    if type(patchSize) is int:
        patchSize = [patchSize, patchSize]
    # Parameters
    I = I.T
    M, N = I.shape
    col_extent = N - patchSize[1] + 1
    row_extent = M - patchSize[0] + 1
    # Get Starting block indices
    start_idx = np.arange(patchSize[0])[:,None]*N + np.arange(patchSize[1])
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(0,row_extent,stride)[:,None]*N + np.arange(0,col_extent,stride)
    # Get all actual indices & index into input array for final output
    idx = start_idx.ravel()[:,None] + offset_idx.ravel()
    out = np.take(I, idx)
    return out


def col2im(Z, patchSize, mm, nn, normalize=True):
    t = np.reshape(np.arange(mm * nn),(mm, nn))
    temp = im2col(t, [patchSize, patchSize]).flatten()
    I = np.bincount(temp, weights=Z.flatten())
    if normalize:
        I /= np.bincount(temp)
    I = np.reshape(I, (mm, nn))
    return I