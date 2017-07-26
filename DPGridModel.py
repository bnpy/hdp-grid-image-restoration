import numpy as np
from numpy.linalg import inv
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from bnpy.ioutil.ModelReader import load_model_at_prefix
from bnpy.suffstats import ParamBag
from util import im2col, col2im


class DPGridModel(object):
    def __init__(self, fileName, **kwargs):
        self.patchModel = load_model_at_prefix(fileName)
        self.patchModelFileName = fileName
        self.calcGlobalParams(**kwargs)

    def calcGlobalParams(self, **kwargs):
        self.D = self.patchModel.obsModel.D
        self.K = self.patchModel.obsModel.K
        self.GP = ParamBag(K=self.K, D=self.D)
        self._calcAllocGP()
        self._calcObsGP()
        self._calcUGP(**kwargs)

    def _calcAllocGP(self):
        # Calculate DP parameters
        logPi = self.patchModel.allocModel.Elogbeta
        self.GP.setField('logPi', logPi, dims='K')

    def _calcObsGP(self):
        # Calculate zero-mean Gaussian parameters
        model = self.patchModel.obsModel
        logdetLam = model.GetCached('E_logdetL', 'all')
        self.GP.setField('logdetLam', logdetLam, dims='K')
        Lam = model.Post.nu[:, np.newaxis, np.newaxis] * inv(model.Post.B)
        self.GP.setField('Lam', Lam, dims=('K', 'D', 'D'))

    def _calcUGP(self, r=0.43, s2=0.21**2):
        self.GP.setField('r', r)
        self.GP.setField('s2', s2)

    def denoise(self, y, sigma, cleanI, T=8):
        self.print_denoising_info(y, cleanI)
        self.PgnPart = self.get_part_info(y)
        betas = self.get_annealing_schedule(sigma, T)
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
            print('PSNR: %.2f dB' % self.calcPSNR(x, cleanI))
        x = self.clip_pixel_intensity(x)
        finalPSNR = float(format(self.calcPSNR(x, cleanI), '.2f'))
        print('Final PSNR: %.2f dB' % finalPSNR)
        return x, finalPSNR

    def print_denoising_info(self, y, cleanI):
        patchSz = int(np.sqrt(self.D))
        print('Pretrained %s: K = %d clusters' % (self.__class__.__name__, self.K))
        print('Patch size: D = %d x %d pixels' % (patchSz, patchSz))
        print('Image size: %d x %d pixels' % y.shape)
        print('PSNR of the noisy image: %.2f dB' % self.calcPSNR(y, cleanI))

    def get_part_info(self, image):
        # Gathers information for partial patches; return a dict
        # whose keys are masks for observable pixels wrt a patch,
        # and values are indices of those pixels wrt the image
        patchSize = int(np.sqrt(self.D))
        H, W = image.shape
        HFull = H + (patchSize - 1) * 2
        WFull = W + (patchSize - 1) * 2
        imgFull = np.reshape(np.arange(HFull * WFull), (HFull, WFull))
        PgnFull = im2col(imgFull, patchSize).T
        NFull = PgnFull.shape[0]
        PgnPart = dict()
        for n in xrange(NFull):
            h, w = np.unravel_index(PgnFull[n], (HFull, WFull))
            hMask = np.logical_and(h >= patchSize - 1, h <= HFull - patchSize)
            wMask = np.logical_and(w >= patchSize - 1, w <= WFull - patchSize)
            mask = np.logical_and(hMask, wMask)
            if not np.all(mask):
                h = h[mask] - (patchSize - 1)
                w = w[mask] - (patchSize - 1)
                idx = np.ravel_multi_index(np.array([h, w]), (H, W))
                if tuple(mask) in PgnPart:
                    PgnPart[tuple(mask)] = np.vstack((PgnPart[tuple(mask)], idx))
                else:
                    PgnPart[tuple(mask)] = np.array([idx])
        return PgnPart

    def get_annealing_schedule(self, sigma, T):
        MINBETA = 0.5/255
        if sigma == MINBETA:
            betas = MINBETA * np.ones(T)
        elif sigma < MINBETA:
            raise ValueError('Noise std shouldn\'t be smaller than %f!' % MINBETA)
        else:
            betaAneal = np.array([sigma])
            tmp = sigma / 2.0
            if tmp > MINBETA and T - len(betaAneal) > 0:
                betaAneal = np.append(betaAneal, np.array([tmp]))
            while tmp / np.sqrt(2) > MINBETA and T - len(betaAneal) > 0:
                tmp /= np.sqrt(2.0)
                betaAneal = np.append(betaAneal, np.array([tmp]))
            if T - len(betaAneal) > 0:
                betaReal = MINBETA * np.ones(T - len(betaAneal))
                betas = np.concatenate((betaAneal, betaReal))
            else:
                betas = betaAneal
        return betas

    def init_x_u_logPi(self, y):
        x = self._initX(y)
        u, uPart = self._initU(y)
        logPi = self._initLogPi()
        return x, u, uPart, logPi

    def _initX(self, y):
        return y

    def _initU(self, y):
        patchSize = int(np.sqrt(self.D))
        patches = im2col(y, patchSize)
        u = np.mean(patches, axis=0)
        uPart = dict()
        for mask, idx in self.PgnPart.items():
            uPart[mask] = np.mean(y.ravel()[idx], axis=1)
        return u, uPart

    def _initLogPi(self):
        return self.GP.logPi

    def update_z(self, beta, logPi, x, u, uPart):
        # fully observable patches
        IP = self.calcIterationParams(beta)
        D, K, GP, patchSize = self.D, self.K, self.GP, int(np.sqrt(self.D))
        Px_minus_u = im2col(x, patchSize) - u
        NFull = Px_minus_u.shape[1]
        resp = np.tile(logPi + 0.5*(IP.logdetSigma + GP.logdetLam), (NFull, 1))
        for k in xrange(K):
            tmp = solve_triangular(beta ** 2 * IP.Rc[k], Px_minus_u,
                                   lower=IP.Rlower[k], check_finite=False)
            resp[:, k] += .5 * np.einsum('dn,dn->n', tmp, tmp)
        resp = np.argmax(resp, axis=1)
        # partially observable patches
        respPart = dict()
        for mask, idx in self.PgnPart.items():
            maskLst = np.array(list(mask), dtype=bool)
            IPPart = self.calcIterationParams(beta, mask=maskLst)
            NPart = idx.shape[0]
            CT_Px_minus_u = np.zeros((D, NPart))
            CT_Px_minus_u[maskLst, :] = x.ravel()[idx].T - uPart[mask]
            this_resp = np.tile(logPi + 0.5*(IPPart.logdetSigma + GP.logdetLam), (NPart, 1))
            for k in xrange(K):
                tmp = solve_triangular(beta ** 2 * IPPart.Rc[k], CT_Px_minus_u,
                                       lower=IPPart.Rlower[k], check_finite=False)
                this_resp[:, k] += .5 * np.einsum('dn,dn->n', tmp, tmp)
            respPart[mask] = np.argmax(this_resp, axis=1)
        return resp, respPart

    def calcIterationParams(self, std, mask=None):
        D, K, GP = self.D, self.K, self.GP
        IP = ParamBag(K=K, D=D)
        if mask is None:
            mask = np.ones(D, dtype=bool)
        invSigma = 1.0 / std ** 2 * np.diag(mask) + GP.Lam
        Rc = np.zeros((K, D, D))
        Rlower = np.ones(K, dtype=bool)
        for k in xrange(K):
            Rc[k], Rlower[k] = cho_factor(invSigma[k], lower=True)
        try:
            IP.setField('Rc', np.tril(Rc), dims=('K', 'D', 'D'))
        except ValueError:
            for k in xrange(K):
                Rc[k] = np.tril(Rc[k])
            IP.setField('Rc', Rc, dims=('K', 'D', 'D'))
        IP.setField('Rlower', Rlower, dims='K')
        logdetSigma = - 2 * np.sum(np.log(np.diagonal(Rc, axis1=1, axis2=2)), axis=1)
        IP.setField('logdetSigma', logdetSigma, dims='K')
        return IP

    def update_v(self, beta, x, u, uPart, resp, respPart):
        # fully observable patches
        IP = self.calcIterationParams(beta)
        D, K, GP, patchSize = self.D, self.K, self.GP, int(np.sqrt(self.D))
        Px_minus_u = im2col(x, patchSize) - u
        NFull = Px_minus_u.shape[1]
        v = np.zeros((NFull, D))
        for k in xrange(K):
            idx_k = np.flatnonzero(resp == k)
            if len(idx_k) == 0:
                continue
            cho = (IP.Rc[k] * beta, bool(IP.Rlower[k]))
            v[idx_k] = cho_solve(cho, Px_minus_u[:, idx_k],
                            overwrite_b=True,
                            check_finite=False).T
        # partially observable patches
        vPart = dict()
        for mask, idx in self.PgnPart.items():
            maskLst = np.array(list(mask), dtype=bool)
            IPPart = self.calcIterationParams(beta, mask=maskLst)
            NPart = len(uPart[mask])
            CT_Px_minus_u = np.zeros((D, NPart))
            CT_Px_minus_u[maskLst, :] = x.ravel()[idx].T - uPart[mask]
            this_v = np.zeros((NPart, D))
            for k in xrange(K):
                idx_k = np.flatnonzero(respPart[mask] == k)
                if len(idx_k) == 0:
                    continue
                cho = (IPPart.Rc[k] * beta, bool(IPPart.Rlower[k]))
                this_v[idx_k] = cho_solve(cho, CT_Px_minus_u[:, idx_k],
                                        overwrite_b=True,
                                        check_finite=False).T
            vPart[mask] = this_v
        return v, vPart

    def update_u(self, beta, x, v, vPart):
        # fully observable patches
        D, GP, patchSize = self.D, self.GP, int(np.sqrt(self.D))
        beta2inv = 1.0 / beta**2
        gamma2 = 1.0 / (1.0 / GP.s2 + D * beta2inv)
        patches = im2col(x, patchSize)
        Px_minus_v = patches.T - v
        u = gamma2 * (GP.r / GP.s2 + beta2inv * np.sum(Px_minus_v, axis=1))
        # partially observable patches
        uPart = dict()
        for mask, idx in self.PgnPart.items():
            maskLst = np.array(list(mask), dtype=bool)
            NPart, DPart = idx.shape
            gamma2 = 1.0 / (1.0 / GP.s2 + DPart * beta2inv)
            Px_minus_v = x.ravel()[idx] - vPart[mask][:, maskLst]
            uPart[mask] = gamma2 * (GP.r / GP.s2 + beta2inv * np.sum(Px_minus_v, axis=1))
        return u, uPart

    def update_x(self, sigma, beta, y, v, vPart, u, uPart):
        D, patchSize = self.D, int(np.sqrt(self.D))
        H, W = y.shape
        def piece_up_patches():
            result = col2im(v.T + u, patchSize, H, W, normalize=False).ravel()
            for mask, idx in self.PgnPart.items():
                maskLst = np.array(list(mask), dtype=bool)
                result += np.bincount(idx.ravel(), minlength=H * W,
                          weights=(uPart[mask][:, np.newaxis]
                                   + vPart[mask][:, maskLst]).ravel())
            result /= D
            return result.reshape(y.shape)
        rec_from_patches = piece_up_patches()
        sigma2, beta2 = sigma ** 2, beta ** 2
        x = (beta2 * y + sigma2 * rec_from_patches) / (sigma2 + beta2)
        return x

    def clip_pixel_intensity(self, image):
        image[image < 0.0] = 0.0
        image[image > 1.0] = 1.0
        return image

    def calcPSNR(self, I, cleanI):
        return 20 * np.log10(1.0 / np.std(cleanI - I))