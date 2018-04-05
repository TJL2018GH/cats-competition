class BaseClassifier:

    def __init__(self, feature_length, num_classes):
        pass

    def train(self, features, labels):
        """
        Using a set of features and labels, trains the classifier and returns the training accuracy.
        :param features: An MxN matrix of features to use in prediction
        :param labels: An N row list of labels to train to predict
        :return: Prediction accuracy, as a float between 0 and 1
        """
        pass

    def predict(self, features, labels):
        """
        Using a set of features and labels, predicts the labels from the features,
        and returns the accuracy of predicted vs actual labels.
        :param features: An MxN matrix of features to use in prediction
        :param labels: An N row list of labels to test prediction accuracy on
        :return: Prediction accuracy, as a float between 0 and 1
        """
        pass


    def reset(self):
        """
        Resets the trained weights / parameters to initial state
        :return:
        """
        pass