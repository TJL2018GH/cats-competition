from feature_selectors.base_selector import BaseSelector
import numpy as np
from scipy import stats


def group_by_classifier(features, labels):
    """
    Divides patient samples into groups according to the corresponding label
    :param features: a matrix of features from patient
    :param labels: a list of corresponding labels
    """

    her2_samples = features[np.where(labels == 'HER2+')]
    hr_samples = features[np.where(labels == 'HR+')]
    trip_neg_samples = features[np.where(labels == 'Triple Neg')]

    return her2_samples, hr_samples, trip_neg_samples


def mannwhitneyu_test(x,y):
    """
    Performs a Mann-Whitney U test on data arrays x and y
    :param x: list of expression data
    :param y: list of expression data
    :return p_value: p value of the test
    """
    try:
        p_value = stats.mannwhitneyu(x, y)[1]
    except ValueError:
        p_value = 1

    return p_value


class MannWhitneySelector(BaseSelector):
    def select_features(self, data, labels):
        """
        Selects interesting features (column indices) from given data matrix using Mann-Whitney U tests
        All the observations from both groups are independent of each other, The responses are ordinal
        (i.e., one can at least say, of any two observations, which is the greater)
        :param data: MxN matrix containing features as columns, and samples as rows
        :param labels: Mx1 matrix containing corresponding data labels
        :return: list of indices of interesting features
        """

        her2_samples, hr_samples, trip_neg_samples = group_by_classifier(data, labels)

        # Pairwise p values of a mann whitney u test, the first row between her2 and hr samples
        # The second between her2 samples and triple negative samples and the third triple negative and hr samples

        p_values = np.zeros((3, data.shape[1]))


        for index in range(data.shape[1]):
            p_values[0, index] = mannwhitneyu_test(her2_samples[:, index], hr_samples[:, index])
            p_values[1, index] = mannwhitneyu_test(her2_samples[:, index], trip_neg_samples[:, index])
            p_values[2, index] = mannwhitneyu_test(trip_neg_samples[:, index], hr_samples[:, index])

        # sorted_indices = sorted(list(set(np.asarray(np.where(p_values < 0.005)[1]))))
        sorted_indices = np.asarray(np.where(p_values == p_values.min())[1])
        return sorted_indices