import SVM
from matplotlib import pyplot as plt
import sklearn.datasets as dataset
from sklearn import preprocessing


def display_char(dataset, char_index):
    plt.ion()
    im = dataset['images']
    char_image = im[char_index, ...]
    plt.imshow(char_image, cmap='binary')
    plt.show()
    plt.waitforbuttonpress()


if __name__ == '__main__':
    '''
    Definition main
    '''
    # Load data using sklearn
    split_point = 100
    digits = dataset.load_digits(2)
    features = digits['data']
    label = digits['target']
    label[label == 1] = 1
    label[label == 0] = -1

    # Display a sample char
    display_char(digits, 10)

    # Standardize Dataset
    features = preprocessing.scale(features)

    # split train and test
    train_features = features[0:split_point, :]
    train_label = label[0:split_point]

    test_features = features[split_point + 1:, :]
    test_label = label[split_point + 1:]

    _, n_feat = features.shape
    # Train SVM
    my_svm = SVM.SVM_Pegasos(n_feat, 10000, 1.0)
    my_svm.train(train_features, train_label)

    # Test it
    acc, result = my_svm.test(test_features, test_label, True)
    print('Accuracy on test set is %1.4f' % acc)
