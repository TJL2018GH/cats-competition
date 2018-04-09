import numpy as np

from feature_selectors.base_selector import BaseSelector


class RandomSelector(BaseSelector):
    def select_features(self, data):
        """
        Selects interesting features (column indices) from given data matrix.
        :param data: MxN matrix containing features as columns, and samples as rows
        :return: list of indices of interesting features
        """
        return np.random.choice(range(0, len(data)), int(len(data) * 0.25))