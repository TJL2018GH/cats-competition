# Decision Tree Classifier
# hand-crafted by Beans

# IMPORTS
from classifiers.base_classifier import BaseClassifier
from numpy import unique
from sklearn import tree


class DecisionTreeClassifier(BaseClassifier):
    def __init__(self, feature_length, num_classes):
        super().__init__(feature_length, num_classes)
        self.num_classes = num_classes

        # model build
        # criterion='entropy' gives split criterion, here: information gain
        self.model = tree.DecisionTreeClassifier(criterion='entropy')

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

        # save depiction of the tree model
        # create .png with bash command: 'dot -Tpng dt_classifier.dot -o dt_classifier.png'
        tree.export_graphviz(self.model, out_file='dt_classifier.dot')

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
