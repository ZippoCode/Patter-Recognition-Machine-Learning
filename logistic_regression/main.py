import numpy as np

from data_io import gaussians_dataset, load_got_dataset
from logistic_regression import LogisticRegression
from visualization import plot_boundary
np.random.seed(191090)

def main():
    """
    Main function
    :return: None
    """


    #x_train, y_train, x_test, y_test = gaussians_dataset(2, [40, 25], [[1, 2], [10, 40]], [[10, 11], [14, 20]])
    x_train, y_train, train_names, x_test, y_test, test_names, feature_names = load_got_dataset(path='data/got.csv', train_split=0.8)

    logistic_regression = LogisticRegression()

    logistic_regression.fit_gradient_descent(x_train, y_train, num_epochs=10000, learning_rate=0.01, verbose=True)

    predictions = logistic_regression.predict(x_test)

    accuracy = float(np.sum(predictions == y_test)) / y_test.shape[0]
    print('Test accuracy: {}'.format(accuracy))

    # Test
    plot_boundary(x_test, test_names, logistic_regression)


if __name__ == '__main__':
    main()
