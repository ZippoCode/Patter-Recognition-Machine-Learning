'''
Class that models a Naive Bayes Classifier
'''

import numpy as np


class NaiveBayesClassifier:
    '''

    '''

    def __init__(self):
        '''
        Class constructor
        '''

        self._classes = None
        self._num_classes = 0

        self._eps = np.finfo(np.float32).eps

        # Array of classes prior probabilities
        self._class_priors = []

        # Array of probabilities of a pixel being active (for each class)
        self._pixel_probs_given_class = []

    def fit(self, X=np.array([]), Y=np.array([])):
        '''
        Computes, for each class, a naive likelyhood model (self._pixel_probs_given_class),
        and a prior probality (self.class_priors).
        Both quantities are estimated from examples X and Y

        Parameters:
        ----------------------------------------------------
        :param X: np.array
            input MNIST digits. Has shape (num_train_sample, h, w)
        :param Y: np.array
            labels for MNIST digits. Has shape (num_train_samples,)
        :return:
            NaN
        '''

        num_train_samples, h, w = X.shape

        self._classes = np.unique(Y)
        self._num_classes = len(self._classes)

        print("Compute priority for each class...")
        # Compute prior and pixel probabilities for each class
        for c_idx, c in enumerate(self._classes):
            # Examples of this class
            x_c = X[Y == c]

            # Prior probability
            prior_c = np.sum(np.uint8(Y == c)) / num_train_samples
            self._class_priors.append(prior_c)

            probs_c = self._estimate_pixel_probabilities(x_c)
            self._pixel_probs_given_class.append(probs_c)

    def predict(self, X=np.array([])):
        '''
        Performs inference on test data.
        Inference is performed according with the Bayes rule:
        P = argmax_T (log(P(X/Y) + log(P(Y)) - log(P(X))

        Parameters:
        ---------------------------
        :param X: np.array
            MNIST test images. Has shapes (num_test_samples, h,w)

        Returns
        ----------------------------
        :return:
            prediction: np.array
            model predictions over X. Has shape (num_test_samples,)
        '''

        num_test_samples, h, w = X.shape

        # Initialize log probabilities of class
        class_log_probs = np.zeros(shape=(num_test_samples, self._num_classes))

        print("Prediction...")
        for c in range(0, self._num_classes):
            # Extract class models
            pix_probs_c = self._pixel_probs_given_class[c]
            prior_c = self._class_priors[c]

            # prior probability of this class
            log_prior_c = np.log(prior_c)

            # Likelyhood of examples given class
            log_likelihood_x = self.get_log_likelyhood_under_mode(X, pix_probs_c)

            # Bayes rule for logarithm
            log_prob_c = log_likelihood_x + log_prior_c

            # Set Class Probability for each test example
            class_log_probs[:, c] = log_prob_c

        predictions = np.argmax(class_log_probs, axis=1)

        return predictions

    @staticmethod
    def _estimate_pixel_probabilities(images=np.array([])):
        '''
        Estimate pixel probabilities from data.

        Parameters
        -------------------------------
        :param images: np.array
            images to estimate pixel probabilities from. Has shape (num_images, h, w)

        Returns
        --------------------------------
        :return: pix_probs: np.array
            probabilities for each pixel of being 1, estimated from images.
            Has shape (h,w)
        '''

        return np.mean(images, axis=0)

    def get_log_likelyhood_under_mode(self, image=np.array([]), model=np.array([])):
        '''
        Return the likelyhood of many images under a certain model.
        Naive:
            the likelyhood of many images is the product of the likelyhood of each pixel.
            or the log.likelyhood of the image is the sum of the log.likelyhood of each pixel.
        Parameters
        ---------------------------------------
        :param image: np.array
                input images. Having shape (num_images, h, w)
        :param model: np.array
                a model of pixel probabilities, having shape (h,w)

        Returns
        ----------------------------------------
        :return: likelyhood: np.array
            the likelyhood of each pixel under the model, having shape (h,w).
        '''

        num_samples = image.shape[0]
        model = np.tile(np.expand_dims(model, axis=0), reps=(num_samples, 1, 1))

        idx_1 = (image == 1)
        idx_0 = (image == 0)

        likelyhood = np.zeros_like(image, dtype=np.float32)
        likelyhood[idx_1] = model[idx_1]
        likelyhood[idx_0] = 1 - model[idx_0]

        log_likelyhood = np.apply_over_axes(np.sum, np.log(likelyhood + self._eps), axes=[1, 2])
        log_likelyhood = np.squeeze(log_likelyhood)

        return log_likelyhood
