from feature_selectors.base_selector import BaseSelector


class AllSelector(BaseSelector):
    def select_features(self, data,labels):
        """
        Selects interesting features (column indices) from given data matrix.
        :param data: MxN matrix containing features as columns, and samples as rows
        :return: list of indices of interesting features
        """
        return list(range(0, len(data)))