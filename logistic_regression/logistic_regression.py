import numpy as np

eps = np.finfo(float).eps


def sigmoid_function(X):
    """
    Element-wise sigmoid function

    Parameters:
    --------------------------------

    :param : np.array
        A numpy array of any shape

    Returns:
    --------------------------------
    :return: np.array
        An array having the same shape of X
    """

    return 1 / (1 + np.exp(-X))


def bce_loss(Y_true=np.array([]), Y_pred=np.array([])):
    """
    The binary cross-entropy loss

    Parameters:
    ---------------------------------
    :param Y_true: np.array
        Real labels in [0,1]. shape = (num_examples, )
    :param Y_pred: np.array
        Prediction labels in [0,1]. Shape = (num_examples,)
    :return:
    """
    return -np.mean(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))


def dev_bce_loss_dw(Y_true=np.array([]), Y_pred=np.array([]), X=np.array([])):
    """
    Derivate of Binary Cross Entropy loss w,r,t weights

    Parameters:
    -------------------------------
    :param Y_true: np.array
        Real labels in [0,1]. Shape = (num_samples,)
    :param Y_pred: np.array
        Predicted labels in [0,1]. Shape = (num_samples,)
    :param X: np.array
        Predicted Data. Shape = (num_samples, num_features).

    Returns:
    -----------------------------
    :return: np.array
        Derivate of Binary Cross Entropy
        Has shape = (num_features, )
    """

    # Number of rows
    N = Y_true.shape[0]
    return -np.dot(X.T, (Y_true - Y_pred)) / N


class LogisticRegression:
    """
    Models a logistic regression classifier
    """

    def __init__(self):
        """
        Constructor method
        """
        # Weights Placeholder
        self._w = None

    def fit_gradient_descent(self, X, Y,
                             num_epochs, learning_rate,
                             verbose=False):
        """

        :param X: np.array
            Data. Shape = (num_examples, num_features)
        :param Y: np.array
            Labels. Shape (num_examples,)
        :param num_epochs: int
            Number of Gradient Update
        :param learning_rate: float
            Step toward the descent
        :param verbose: bool
            Whether or not to print the value to cost function
        :return:
        """

        num_samples, num_features = X.shape

        # Initialize randomly weights
        self._w = np.random.normal(loc=0, scale=0.001, size=(num_features,))

        # Loop over epochs
        for e in range(0, num_epochs):

            # Predict training Data
            current_prediction = sigmoid_function(np.dot(X, self._w))

            # Compute (and print) cost
            current_loss = bce_loss(Y_true=Y, Y_pred=current_prediction)
            if verbose and e % 1000 == 0:
                print("Binary Cross Entropy: ", current_loss)

            # Update Weights following gradient
            self._w -= learning_rate * dev_bce_loss_dw(Y_true=Y, Y_pred=current_prediction, X=X)

    def predict(self, X):
        """
        Function that predict.

        Parameters:
        ------------------------------------------
        :param X: np.array
            Data to be predicted. Shape = (num_test_examples, num_features)

        Returns:
        --------------------------------------------
        :return: np.array
            Prediction in [0,1]
            Shape is (num_test_examples,)
        """
        # Predict testing data
        return np.round(sigmoid_function(np.dot(X, self._w)))
