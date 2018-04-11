from sklearn.naive_bayes import GaussianNB
from scipy.stats import rankdata
import numpy as np
from numpy import unique
from feature_selectors.base_selector import BaseSelector





class RFESelector(BaseSelector):

    def select_features(self,data,labels):
        """
        Selects interesting features (column indices) from given data matrix.
        :param data: MxN matrix containing features as columns, and samples as rows
        :return: list of indices of interesting features
        """
        self.class_num = 3
        self.sel_features = 1
        step = 1
        index_features = np.array(range(len(data[0])))
        model = GaussianNB()
        labels = self.labels_to_categorical(labels)
        index_features = self.eliminate(model,data,labels,self.sel_features,self.class_num,step,index_features,0)

        return index_features

    def labels_to_categorical(self,labels):
        _,IDs = unique(labels,return_inverse=True)
        return IDs

    def eliminate(self,model_classifier,data,labels,n_features,class_num,step,index_records,j):
        length= len(data[0])
        for k in range(length - 1*step):
            ranks = []
            model = model_classifier
            model.fit(data,labels)
            features_mean = model.theta_

            j += 1
            for i in range(class_num):
                ranks.append(rankdata(features_mean[i]))

            ranks = np.array(ranks)

            sum_ranks = np.sum(ranks,axis=0)
            ord_sum_ranks = np.argsort(sum_ranks)
            if (len(data[0]) - step) == n_features:

                ind = np.where(np.max(ord_sum_ranks))

            else:

                ind = np.where( ord_sum_ranks < (len(data[0]) - step))

            nsamples,nx,ny = data[:,ind].shape
            data = data[:,ind].reshape((nsamples,nx * ny))
            index_records = index_records[ind]

        return index_records




