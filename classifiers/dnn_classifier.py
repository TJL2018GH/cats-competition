from keras import Sequential
from keras.layers import Dense,Dropout
from keras.utils import to_categorical
from numpy import unique
from sklearn.preprocessing import LabelBinarizer

from classifiers.base_classifier import BaseClassifier


class DeepNeuralClassifier(BaseClassifier):
    encoder = LabelBinarizer()  # please annotate what this is doing (hypothesis: for to_categorical())

    def __init__(self,feature_length,num_classes):
        super().__init__(feature_length,num_classes)
        self.num_classes = num_classes

        # From Keras examples (https://keras.io/getting-started/sequential-model-guide/)
        self.model = Sequential()
        self.model.add(Dense(64,activation='relu',input_dim=feature_length))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64,activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes,activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])

        self.initial_weights = self.model.get_weights()

    def train(self,features,labels):
        """
        Using a set of features and labels, trains the classifier and returns the training accuracy.
        :param features: An MxN matrix of features to use in prediction
        :param labels: An M row list of labels to train to predict
        :return: Prediction accuracy, as a float between 0 and 1
        """
        labels = self.labels_to_categorical(labels)
        result = self.model.fit(features,labels,epochs=16,verbose=0)
        return result.history['acc'][-1]

        # make sure you save model using the same library as we used in machine learning price-predictor

    def predict(self,features,labels):
        """
        Using a set of features and labels, predicts the labels from the features,
        and returns the accuracy of predicted vs actual labels.
        :param features: An MxN matrix of features to use in prediction
        :param labels: An M row list of labels to test prediction accuracy on
        :return: Prediction accuracy, as a float between 0 and 1
        """
        labels = self.labels_to_categorical(labels)
        accuracy = self.model.evaluate(features,labels,verbose=0)[1]
        return accuracy


    def get_prediction(self,features):
        return self.model.predict(features)

    def reset(self):
        """
        Resets the trained weights / parameters to initial state
        :return:
        """
        self.model.set_weights(self.initial_weights)
        pass

    def labels_to_categorical(self,labels):
        _,IDs = unique(labels,return_inverse=True)
        return to_categorical(IDs,num_classes=self.num_classes)
