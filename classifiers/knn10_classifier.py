# K-Nearest-Neighbors Classifier
# hand-crafted by Beans

# IMPORTS
from classifiers.base_classifier import BaseClassifier
from numpy import unique
from sklearn.neighbors import KNeighborsClassifier


class K10NearestNeighborsClassifier(BaseClassifier):
    K = 10 # (default=5) hyper-parameter

    def __init__(self, feature_length, num_classes):
        super().__init__(feature_length, num_classes)