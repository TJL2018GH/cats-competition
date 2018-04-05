# IMPORTS

# NOT YET IMPLEMENTED
# Please announce if you implement it.
# Ultima ratio: Ben provides himself for implementation

from classifiers.base_classifier import BaseClassifier


class KNearestNeighborsClassifier(BaseClassifier):
    # encoder = LabelBinarizer()

    def __init__(self, feature_length, num_classes):
        super().__init__(feature_length, num_classes)
        self.num_classes = num_classes

        ###
        # BUILD YOUR MODEL

        ###

    def train(self, features, labels):
        """
        Using a set of features and labels, trains the classifier and returns the training accuracy.
        :param features: An MxN matrix of features to use in prediction
        :param labels: An M row list of labels to train to predict
        :return: Prediction accuracy, as a float between 0 and 1
        """
        labels = self.labels_to_categorical(labels)
        # result =
        return

    def predict(self, features, labels):
        """
        Using a set of features and labels, predicts the labels from the features,
        and returns the accuracy of predicted vs actual labels.
        :param features: An MxN matrix of features to use in prediction
        :param labels: An M row list of labels to test prediction accuracy on
        :return: Prediction accuracy, as a float between 0 and 1
        """
        labels = self.labels_to_categorical(labels)
        # accuracy =
        return

    def labels_to_categorical(self, labels):
        _, IDs = unique(labels, return_inverse=True)
        return to_categorical(IDs, num_classes=self.num_classes)