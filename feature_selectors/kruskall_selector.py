from feature_selectors.base_selector import BaseSelector
from scipy import stats
import numpy as np


def group_by_classifier(features, labels):
    """
    Divides patient samples into groups according to the corresponding label
    :param features: a matrix of features from patient
    :param labels: a list of corresponding labels
    :return feature matrices of the categorical data
    """

    her2_samples = features[np.where(labels == 'HER2+')]
    hr_samples = features[np.where(labels == 'HR+')]
    trip_neg_samples = features[np.where(labels == 'Triple Neg')]

    return her2_samples, hr_samples, trip_neg_samples


class KruskallSelector(BaseSelector):
    def select_features(self, data, labels):
        """
        Selects interesting features (column indices) from given data matrix using the K-W test
        This test assumes that the compared groups have the same distribution
        :param data: MxN matrix containing features as columns, and samples as rows
        :param labels: Mx1 matrix containing corresponding data labels
        :return: list of indices of interesting features
        """

        her2_samples, hr_samples, trip_neg_samples = group_by_classifier(data, labels)

        p_values = np.zeros((data.shape[1]))
        for index in range(data.shape[1]):
            try:
                p_values[index] = \
                stats.kruskal(her2_samples[:, index], hr_samples[:, index], trip_neg_samples[:, index])[1]
            except ValueError:
                p_values[index]=1

        # Multiple testing correction provide no significant variables, we'll stick with this for now
        #significant_p_value_indices = np.asarray(np.where(np.array(p_values) < 0.03))[0]
        significant_p_value_indices = np.asarray(np.where(p_values == p_values.min())[0])
        print(significant_p_value_indices)

        return significant_p_value_indices
