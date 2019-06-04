import numpy as np
import pylab as plot


def Hbeta(D=np.array([]), beta=1.0):
    '''
    Compute the perplexity and the P-row for a specific value of the precision of
    Gaussian distribution

    :param D:
    :param beta:
    :return:
    '''

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    '''
    Perform a binary search to get P-values in such a way that each conditional
    Gaussian has the same perplexity

    :param D:
    :param beta:
    :return:
    '''
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point ", i, " of ", n, "...")

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = - np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # if not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the value
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = tries + 1

        # set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: ", np.mean(np.square(1 / beta)))
    return P


def pca(X=np.array([]), num_dims=50):
    '''
    Runs PCA on the NxD array X in order to reduce its dimensionality
    to num_dims dimensions

    :param X:
    :param num_dims:
    :return:
    '''
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    matrix_cov = np.dot(X.T, X)
    (l, M) = np.linalg.eig(matrix_cov)
    Y = np.dot(X, M[:, 0:num_dims])
    return Y


def t_sne(X=np.array([]), num_dims=2, initial_dims=50, perplexity=30.0):
    '''

    :param X:
    :param num_dims:
    :param initial_dims:
    :param perplexity:
    :return:
    '''

    # check inputs
    if isinstance(num_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(num_dims) != num_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variavles
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 100
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, num_dims)
    dY = np.zeros((n, num_dims))
    iY = np.zeros((n, num_dims))
    gains = np.ones((n, num_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4  # early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (num_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration ", (iter + 1), ": error is", C)

        # Stop lying about P-values
        if iter == 100:
            P = P / 4

    # Return solution
    return Y


if __name__ == '__main__':
    print("Run Y = t_sne.t_sne(X, num_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("file/mnist2500_X.txt")
    labels = np.loadtxt("file/mnist2500_labels.txt")
    Y = t_sne(X, 2, 50, 20.0)
    plot.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plot.show()
