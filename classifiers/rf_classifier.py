# Random Forest Classifier (L. Breiman, “Random Forests”, Machine Learning, 45(1), 5 - 32, 2001)
# hand-crafted by Beans

# IMPORTS
from classifiers.base_classifier import BaseClassifier
from numpy import unique
from sklearn.ensemble import RandomForestClassifier


class RForestClassfier(BaseClassifier):
    def __init__(self, feature_length, num_classes):
        super().__init__(feature_length, num_classes)
        self.num_classes = num_classes

        # model build
        # max_depth=5 (a mode of regularization)
        self.model = RandomForestClassifier(n_estimators=10, criterion='entropy')

    def train(self, features, labels):
        """
        Using a set of features and labels, trains the classifier and returns the training accuracy.
        :param features: An MxN matrix of features to use in prediction
        :param labels: An M row list of labels to train to predict
        :return: Prediction accuracy, as a float between 0 and 1
        """
        labels = self.labels_to_categorical(labels)
        self.model.fit(features, labels)
        accuracy = self.model.score(features, labels)
        return accuracy

    def predict(self, features, labels):
        """
        Using a set of features and labels, predicts the labels from the features,
        and returns the accuracy of predicted vs actual labels.
        :param features: An MxN matrix of features to use in prediction
        :param labels: An M row list of labels to test prediction accuracy on
        :return: Prediction accuracy, as a float between 0 and 1
        """
        labels = self.labels_to_categorical(labels)
        accuracy = self.model.score(features, labels)
        return accuracy

    def get_prediction(self,features):
        return self.model.predict(features)

    def labels_to_categorical(self, labels):
        _, IDs = unique(labels, return_inverse=True)
        return IDs
