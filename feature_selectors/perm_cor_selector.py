# permutation test followed by correlation elimination (PermCor, BeanSel) to retrieve k features
# hand-crafted by Beans

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

def cross_difference(a, b, c):
    """
    Returns 'cross-difference' (self-explanatory). Assumes equal class importance, which is a valid assumption.
    :param a: vector of ints
    :param b: vector of ints
    :param c: vector of ints
    :return:
    """
    a, b, c = np.mean(a), np.mean(b), np.mean(c)
    return (abs(a-b) + abs(a-c) + abs(b-c)) / 3

def permutation_test(her2, hr, trip_neg):
    """
    Determines significance of the test statistics 'cross-difference' by permutation test.
    :param her2: values for a column of class her2
    :param hr: values for a column of class hr
    :param trip_neg: values for a column of class triple_neg
    :return: signficance of the test statistic
    """
    len_her2, len_hr, len_trip_neg = len(her2), len(hr), len(trip_neg)
    native_cross_difference = cross_difference(her2, hr, trip_neg)
    collective = np.concatenate((her2, hr, trip_neg), axis=0)
    truth_vector = [] # if random_cross_difference > native_cross_difference True, else False is appended
    for i in range(100):
        np.random.shuffle(collective)
        her2, hr, trip_neg = collective[0:len_her2], collective[len_her2:len_her2+len_hr], collective[-len_trip_neg:]
        random_cross_difference = cross_difference(her2, hr, trip_neg)
        if random_cross_difference > native_cross_difference:
            truth_vector.append(True)
        else:
            truth_vector.append(False)
    return sum(truth_vector) / len(truth_vector)


class PermCorSelector(BaseSelector):
    def select_features(self, data, labels):
        """
        Selects interesting features (column indices) from given data matrix using the PermCor approach.
        This includes a permutation test to assess the significance of the cross-distance and rank the features.
        This is followed by iterative removal of features with cross-correlation above a certain cutoff.
        Retrieved are only k features that rank highest according to the above scheme.
        No distribution assumptions are enforced onto the data.
        :param data: MxN matrix containing features as columns, and samples as rows
        :param labels: Mx1 matrix containing corresponding data labels
        :return: list of indices of interesting features
        """
        her2_samples, hr_samples, trip_neg_samples = group_by_classifier(data, labels)

        p_values = np.zeros((data.shape[1]))
        for index in range(data.shape[1]):
            print(index)
            p_values[index] = permutation_test(her2_samples[:, index], hr_samples[:, index], trip_neg_samples[:, index])
        indices_sorted = np.argsort(p_values)

        return indices_sorted[1:10]