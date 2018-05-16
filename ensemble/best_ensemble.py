from numpy import unique
import pandas as pd
from classifiers.base_classifier import BaseClassifier
from classifiers.dnn_classifier import DeepNeuralClassifier
from classifiers.dt_classifier import DecisionTreeClassifier
from classifiers.linsvc import SupportVectorMachineLinearKernelOneVsRestClassifier
from classifiers.nm_classifier import NearestMeanClassifier
from classifiers.knn_classifier import KNearestNeighborsClassifier
from classifiers.nvb_classifier import NaiveBayesClassifier
from classifiers.rf_classifier import RForestClassfier
from classifiers.svmlin_classifier import SupportVectorMachineLinearKernelClassifier
from classifiers.svmpol_classifier import SupportVectorMachinePolynomialKernelClassifier
from classifiers.svmrbf_classifier import SupportVectorMachineRbfKernelClassifier
from classifiers.gba_classifier import GradientBoostingAlgorithm
from classifiers.lr_classifier import LogisticRegressionClassifier
from sklearn.metrics import accuracy_score


class BestEnsemble(BaseClassifier):
    '''
    this class take the best combinations of feature and model
    can build the models and generate "hard" voting prediction
    '''
    def __init__(self, feature_length, num_classes, x=10):

        super().__init__(feature_length, num_classes)

        self.combinations = []
        self.models = []
        self.num_classes = num_classes


    def add_combination(self, combination):
        '''
        add to or update the combination attribute with a dictonary storing all the information
        :param combination: dictionary storing all the information of the combination

        '''
        self.combinations.append(combination)

    def add_combinations_list(self, combinations):
        '''
        add combinations to combinations attribute from a list of dictonary and exclude itself from the list
        :param combinations: list of dictionary
        '''
        for model_combination in combinations:
            if model_combination['model_name'] != 'best_ens':
                self.combinations.append(model_combination)

    def clean_model(self):
        '''
        clean the attributes models
        and combination and remove duplicate entries to prevent slowdown of ensemble over time
        '''
        self.models = []

        # removing duplicate
        keys = []
        cleaned = []
        for combination in self.combinations:
            name = combination['model'].__name__
            features = str(combination['indices'])
            key = name + features
            if key not in keys:
                keys.append(key)
                cleaned.append(combination)

        self.combinations = cleaned

    def vote(self,models):
        '''
        take the prediction of each combination and create prediction
        assigning the class of a sample by computing the probability(frequency)
        :param models: the list of dictionary storing all the information of each selected combination selector-model
        :return: voted prediction
        '''

        #initialising list and count
        predictions = []
        final_prediction = []
        tot_model = 0

        #retrieve the prediction and store in a list
        for models_set in models:
            tot_model += 1

            predictions.append(pd.DataFrame(models_set['prediction']))

        df_prediction = predictions[0]

        # create a dataframe
        for i in range(1, len(predictions)):
            df_prediction = df_prediction.join(predictions[i], lsuffix='_caller', rsuffix='_other')

        #count and compute frequency of samples labels and compute the "hard votin"
        for _, row in df_prediction.iterrows():
            uni, counts = unique(row, return_counts=True)

            max_temp = 0
            idx_temp = 0
            for i in range(len(uni)):

                if counts[i] / tot_model > max_temp:
                    idx_temp = i
            final_prediction.append(uni[idx_temp])

        return final_prediction

    def train(self, features, labels):
        """
        Using a set of features and labels, trains the classifier and returns the training accuracy.
        :param features: An MxN matrix of features to use in prediction
        :param labels: An M row list of labels to train to predict
        :return: Prediction accuracy, as a float between 0 and 1
        """
        self.clean_model()
        labels = self.labels_to_categorical(labels)
        for combination in self.combinations:
            model = combination['model'](len(combination['indices']), self.num_classes)
            model.train(features[:, combination['indices']], labels)
            pred_responses = model.get_prediction(features[:, combination['indices']])
            self.models.append({'model': model, 'info_comb': combination, 'prediction': pred_responses})

        pred_responses = self.vote(self.models)

        accuracy = accuracy_score(labels, pred_responses)

        return accuracy

    # make sure you save model using the same library as we used in machine learning price-predictor

    def predict(self, features, labels):
        """
        Using a set of features and labels, predicts the labels from the features,
        and returns the accuracy of predicted vs actual labels.
        :param features: An MxN matrix of features to use in prediction
        :param labels: An M row list of labels to test prediction accuracy on
        :return: Prediction accuracy, as a float between 0 and 1
        """
        label_test = self.labels_to_categorical(labels)
        pred_responses = self.get_prediction(features)
        accuracy = accuracy_score(label_test, pred_responses)
        return accuracy

    def get_prediction(self,features):
        '''
        this function get the prediction from the
        :param features: sample to predict
        :return: the final prediction from the "hard" voting function
        '''
        # retrieve the models and store prediction in a list
        # every models uses the features selected in the middle fold of the cross validation
        for i in range(len(self.models)):
            combination = self.models[i]['info_comb']
            self.models[i]['prediction'] = self.models[i]['model'].get_prediction(features[:, combination['indices']])

        return self.vote(self.models)

    def labels_to_categorical(self, labels):
        '''
        convert the labels from string to number
        :param labels: labels list of string
        :return: labels converted in number
        '''
        _, IDs = unique(labels, return_inverse=True)
        return IDs
