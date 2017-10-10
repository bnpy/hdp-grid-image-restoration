import numpy as np
from PIL import Image


def imread(fileName, outputFormat='L'):
    '''
        Read in an image, and return its normalized pixel values (gray scale by default)
    '''
    im = Image.open(fileName)
    W, H = im.size[:2]
    try:
        im = im.convert(outputFormat)
    except:
        raise TypeError('Unable to convert image %s from %s to %s!' % (fileName, im.mode, outputFormat))
    imPix = im.load()
    if outputFormat == 'L':
        result = np.zeros((H, W), dtype=np.float64)
    else:
        assert outputFormat == 'YCbCr'
        result = np.zeros((H, W, 3), dtype=np.float64)
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


def ycbcr2rgb(image):
    H, W, C = image.shape
    assert C == 3
    x = Image.fromarray(np.uint8(np.round(255 * image)),
              mode='YCbCr').convert('RGB').load()
    result = np.zeros(image.shape)
    for h in xrange(H):
        for w in xrange(W):
            result[h, w] = x[w, h]
    return result / 255