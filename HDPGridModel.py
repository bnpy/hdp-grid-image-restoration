import numpy as np
from scipy.io import loadmat
from scipy.special import psi
from bnpy.allocmodel.topics.LocalStepManyDocs import updateLPGivenDocTopicCount
from bnpy.data import GroupXData
from bnpy.init.FromScratchBregman import runKMeans_BregmanDiv
from bnpy.ioutil.SuffStatBagIO import loadSuffStatBag
from bnpy.obsmodel import ZeroMeanGaussObsModel
from bnpy.util import EPS
from util import im2col, col2im, ycbcr2rgb
from DPGridModel import DPGridModel


class HDPGridModel(DPGridModel):
    def __init__(self, *args, **kwargs):
        super(HDPGridModel, self).__init__(*args, **kwargs)

    def _calcAllocGP(self):
        # calculate HDP parameters
        model = self.patchModel.allocModel
        self.GP.setField('alphaPi0', model.alpha_E_beta(), dims='K')
        self.GP.setField('alphaPi0Rem', model.alpha_E_beta_rem())

    def _initLogPi(self):
        K = self.K
        return np.log(1.0 / K) * np.ones(K)

    def denoise(self, y, sigma, cleanI, T=8, **kwargs):
        self.print_denoising_info(y, cleanI)
        self.PgnPart = self.get_part_info(y)
        betas = self.get_annealing_schedule(sigma, T)
        self.train_image_specific_topics(y, sigma, **kwargs)
        x, u, uPart, logPi = self.init_x_u_logPi(y)
        for t in xrange(T):
            print('Iteration %d/%d' % (t + 1, T))
            beta = betas[t]
            print('updating z...')
            resp, respPart = self.update_z(beta, logPi, x, u, uPart)
            print('updating v...')
            v, vPart = self.update_v(beta, x, u, uPart, resp, respPart)
            print('updating u...')
            u, uPart = self.update_u(beta, x, v, vPart)
            print('updating x...')
            x = self.update_x(sigma, beta, y, v, vPart, u, uPart)
            print('updating pi...')
            logPi = self.update_pi(self.get_N(resp, respPart))
            print('PSNR: %.2f dB' % self.calcPSNR(x, cleanI))
        x = self.clip_pixel_intensity(x)
        finalPSNR = float(format(self.calcPSNR(x, cleanI), '.2f'))
        print('Final PSNR: %.2f dB' % finalPSNR)
        return x, finalPSNR

    def update_pi(self, Nk):
        theta = self.GP.alphaPi0 + Nk / self.D
        logPi = psi(theta) - psi(np.sum(theta) + self.GP.alphaPi0Rem)
        return logPi

    def train_image_specific_topics(self, y, sigma, Niter=50, Kfresh=100, pixelMask=None):
        print('Training %d image-specific clusters...' % Kfresh)
        D, patchSize, GP = self.D, int(np.sqrt(self.D)), self.GP
        # gather fully observable patches
        if pixelMask is None:  # gray-scale image denoising
            v = im2col(y, patchSize)
        else:  # color image inpainting
            C = 3
            patchMask = np.logical_not(np.any(im2col(pixelMask, patchSize), axis=0))
            v = np.hstack(tuple([im2col(y[:, :, c], patchSize)[:, patchMask] for c in xrange(C)]))
        v -= np.mean(v, axis=0)
        v = v.T
        testData = GroupXData(X=v, doc_range=[0, len(v)], nDocTotal=1)
        testData.name = 'test_image_patches'
        # set up hyper-parameters and run Bregman k-means
        cached_B_name = 'models/HDP/B.mat'
        xBar = loadmat(cached_B_name)['Cov']
        xBar2 = loadmat(cached_B_name)['Cov2']
        tmp0 = (np.diag(xBar) + sigma**2)**2
        tmp1 = np.diag(xBar2) + 6 * np.diag(xBar) * sigma**2 + 3 * sigma**4
        nu = D + 3 + 2 * np.sum(tmp0) / np.sum(tmp1 - tmp0)
        B = (nu - D - 1) * (xBar + sigma**2 * np.eye(D))
        obsModel = ZeroMeanGaussObsModel(D=D, min_covar=1e-8, inferType='memoVB', B=B, nu=nu)
        Z, Mu, Lscores = runKMeans_BregmanDiv(testData.X, Kfresh, obsModel,
                                              Niter=Niter, assert_monotonic=False)
        Korig = self.K
        Kall = np.max(Z) + Korig + 1
        Kfresh = Kall - Korig
        Z += Korig
        # load SuffStats of training images
        trainSS = loadSuffStatBag('models/HDP/SS.dump')
        trainSS.insertEmptyComps(Kfresh)
        # construct SuffStats of the test image
        DocTopicCount = np.bincount(Z, minlength=int(Kall)).reshape((1, Kall))
        DocTopicCount = np.array(DocTopicCount, dtype=np.float64)
        resp = np.zeros((len(Z), Kall))
        resp[np.arange(len(Z)), Z] = 1.0
        testLP = dict(resp=resp, DocTopicCount=DocTopicCount)
        alphaPi0 = np.hstack((GP.alphaPi0, GP.alphaPi0Rem / (Kfresh+1) * np.ones(Kfresh)))
        alphaPi0Rem = GP.alphaPi0Rem / (Kfresh+1)
        testLP = updateLPGivenDocTopicCount(testLP, DocTopicCount, alphaPi0, alphaPi0Rem)
        testSS = self.patchModel.get_global_suff_stats(testData, testLP,
                                 doPrecompEntropy=1, doTrackTruncationGrowth=1)
        xxT = np.zeros((Kall, D, D))
        for k in xrange(Korig, Kall):
            idx = Z == k
            tmp = np.einsum('nd,ne->de', v[idx], v[idx])
            tmp -= testSS.N[k] * sigma**2 * np.eye(D)
            val, vec = np.linalg.eig(tmp)
            val[val < EPS] = EPS
            xxT[k] = np.dot(vec, np.dot(np.diag(val), vec.T))
        testSS.setField('xxT', xxT, dims=('K', 'D', 'D'))
        testSS.setUIDs(trainSS.uids)
        # combine training and test SS; update model parameters
        combinedSS = trainSS + testSS
        self.patchModel.update_global_params(combinedSS)
        self.calcGlobalParams()

    def get_N(self, resp, respPart):
        N = np.bincount(resp, minlength=self.K)
        for mask in self.PgnPart.keys():
            N += np.bincount(respPart[mask], minlength=self.K)
        N = np.array(N, dtype=np.float64)
        return N

    def inpaint(self, y, pixelMask, T=20, **kwargs):
        self.print_inpainting_info(y)
        betas = 1.0 / np.sqrt(10 * np.array([1, 2, 16, 128, 512]))
        D, patchSize = self.D, int(np.sqrt(self.D))
        if np.any(pixelMask[:patchSize]) or np.any(pixelMask[:, :patchSize]) or \
            np.any(pixelMask[-patchSize+1:]) or np.any(pixelMask[:, -patchSize+1:]):
            raise ValueError('The current implementation does not support inpainting boundary pixels!')
        self.PgnPart = dict()
        self.train_image_specific_topics(y, betas[-1], pixelMask=pixelMask, **kwargs)
        mask_unseen = np.any(im2col(pixelMask, patchSize), axis=0)
        mask_seen = np.logical_not(mask_unseen)
        result = y.copy()
        result[pixelMask] = self.GP.r
        if y.ndim == 3:
            C = 3
        else:
            raise TypeError('The current implementation only supports color-image inpainting!')
        for c in xrange(C):
            print('Inpainting channel %d/%d...' % (c+1, C))
            x, u, uPart, logPi = self.init_x_u_logPi(result[:, :, c])
            resp_seen, respPart = self.update_z(betas[-1], logPi, x, u, uPart, patchLst=mask_seen)
            v_seen, vPart = self.update_v(betas[-1], x, u, uPart, resp_seen, respPart, patchLst=mask_seen)
            u_seen, uPart = self.update_u(betas[-1], x, v_seen, vPart, patchLst=mask_seen)
            for i, beta in enumerate(betas):
                print('  beta value %d/%d' % (i+1, len(betas)))
                IP = self.calcIterationParams(beta)
                for t in xrange(T):
                    resp_unseen, respPart = self.update_z(beta, logPi, x, u, uPart,
                                                          patchLst=mask_unseen, IP=IP)
                    v_unseen, vPart = self.update_v(beta, x, u, uPart, resp_unseen, respPart,
                                                    patchLst=mask_unseen, IP=IP)
                    u_unseen, uPart = self.update_u(beta, x, v_unseen, vPart,
                                                    patchLst=mask_unseen)
                    logPi = self.update_pi(self.get_N(np.concatenate((resp_seen, resp_unseen)), respPart))
                    NFull = len(mask_seen)
                    v, u = np.zeros((NFull, D)), np.zeros(NFull)
                    v[mask_seen] = v_seen
                    v[mask_unseen] = v_unseen
                    u[mask_seen] = u_seen
                    u[mask_unseen] = u_unseen
                    x = self.update_x_by_inpainting(result[:, :, c], v, u, pixelMask)
                    print '    inner iteration %d/%d' % (t+1, T)
            result[:, :, c] = x
        result = ycbcr2rgb(self.clip_pixel_intensity(result))
        return result

    def print_inpainting_info(self, y):
        patchSz = int(np.sqrt(self.D))
        print('Pretrained %s: K = %d clusters' % (self.__class__.__name__, self.K))
        print('Patch size: D = %d x %d pixels' % (patchSz, patchSz))
        print('Image size: %d x %d pixels' % y.shape[:2])

    def update_x_by_inpainting(self, y, v, u, pixelMask):
        H, W = y.shape
        patchSize = int(np.sqrt(self.D))
        rec_from_patches = col2im(v.T+u, patchSize, H, W)
        x = np.zeros((H, W))
        x[~pixelMask] = y[~pixelMask]
        x[pixelMask] = rec_from_patches[pixelMask]
        return x