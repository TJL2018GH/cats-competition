class BaseSelector:

    def select_features(self, data, labels):
        """
        Selects interesting features (column indices) from given data matrix.
        :param data: MxN matrix containing features as columns, and samples as rows
        :param labels: Mx1 matrix containing corresponding data labels
        :return: list of indices of interesting features
        """
        pass