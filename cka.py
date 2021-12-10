import math
import numpy as np


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.
    Args:
        x: A num_examples x num_features matrix of features.
    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    return x.dot(x.T)

class IncrementalCKA:
    def __init__(self, num_layers_0: int, num_layers_1: int) -> None:
        """

        Args:
            num_layers_0: the number of layers in a network 0.
            num_layers_1: the number of layers in a network 1.
        """
        if num_layers_0 < 1:
            raise ValueError(f"`num_layers_0` should be positive: {num_layers_0}")


        if num_layers_1 < 1:
            raise ValueError(f"`num_layers_1` should be positive: {num_layers_1}")


        self.num_layers_0 = num_layers_0
        self.num_layers_1 = num_layers_1
        self._num_mini_batches = np.zeros((num_layers_0, num_layers_1), dtype=int)  # K in the paper
        self._kl = np.zeros((num_layers_0, num_layers_1))
        self._kk = np.zeros(num_layers_0)
        self._ll = np.zeros(num_layers_1)


    @staticmethod
    def _hsic(K: np.ndarray, L: np.ndarray) -> float:
        """
        Eq. 3

        Args:
            K: gram matrix. The shape shape is N \times N, where N is the number of samples in a mini-batches.
            L: gram matrix. The shape shape is N \times N.

        Note that the diagonal elements are zero.

        Returns: HSIC_1 value between K and L.

        """
        #print(K.shape,L.shape)
        n = K.shape[0]
        first = np.trace(np.matmul(K, L))
        second = np.sum(K) * np.sum(L) / (n - 1) / (n - 2)
        third = 2. / (n - 2) * np.sum(K, axis=0).dot(np.sum(L, axis=0))
        denom = n * (n - 3)
        return 1. / denom * (first + second - third)


    def increment_cka_score(self, index_feature_x: int, index_feature_y: int, features_x: np.ndarray,
                            features_y: np.ndarray) -> None:
        """
        Update cka score between `index_feature_x` and `index_feature_y` using a mini-batch.
        This function computes HISC_1 values defined by Eq. 3 and stores them rather than returns them.

        Args:
            index_feature_x: the index for layer in a model 0.
            index_feature_y: the index for layer in a model 1.
            features_x: feature representation extracted by the model 0. The shape is N \times n_features.
            features_y: feature representation extracted by the model 1. The shape is N \times n_features.

            Note that
                - the numbers of samples of `features_x` and `features_y` should be the same.
                - the dimensionalities of `features_x` and `features_y` can differ.

        Returns:
            None.

        """
        assert 0 <= index_feature_x <= self.num_layers_0 - 1
        assert 0 <= index_feature_y <= self.num_layers_1 - 1


        self._num_mini_batches[index_feature_x, index_feature_y] += 1


        gram_x = gram_linear(features_x)
        gram_y = gram_linear(features_y)
        np.fill_diagonal(gram_x, 0)
        np.fill_diagonal(gram_y, 0)


        # since computing terms used in the denom of cka can be skipped,
        # we only compute them when either layer's index is 0.
        if index_feature_x == 0 or index_feature_y == 0:
            if index_feature_x == 0:
                self._ll[index_feature_y] += self._hsic(gram_y, gram_y)
            if index_feature_y == 0:
                self._kk[index_feature_x] += self._hsic(gram_x, gram_x)


        self._kl[index_feature_x, index_feature_y] += self._hsic(gram_x, gram_y)


    def cka(self) -> np.ndarray:
        """
        Compute mini-batch CKA defined by Eq. 2 in the original paper.

        Returns: `np.darray` whose element is cka. The shape is `num_layers_0` \times `num_layers_1`.

        """
        K = np.min(self._num_mini_batches)


        assert K == np.max(self._num_mini_batches)


        cka_score = np.zeros(self._kl.shape)
        for l0, kk in enumerate(self._kk):
            kk = np.sqrt(kk / K)
            for l1, ll in enumerate(self._ll):
                denom = kk * np.sqrt(ll / K)
                cka_score[l0, l1] = self._kl[l0, l1] / K / denom
        return cka_score
