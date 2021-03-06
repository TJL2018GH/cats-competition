from classifiers.base_classifier import BaseClassifier
from sklearn.ensemble.voting_classifier import VotingClassifier
from numpy import unique
from classifiers.base_classifier import BaseClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

class VotingEnsemble(BaseClassifier):

	def __init__(self,feature_length,num_classes,x=10):

		super().__init__(feature_length,num_classes)


		self.model = VotingClassifier(
			  estimators=[
			  ('gba', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)),
			  ('knn', KNeighborsClassifier(metric='manhattan', weights='distance', n_neighbors=3)),
			  ('Nc', NearestCentroid(metric='manhattan')),('nvb',GaussianNB()),
			  ('rf', RandomForestClassifier(n_estimators=10,criterion='entropy')),
			  ('svmlin', svm.SVC(kernel='linear')),('svmpol', svm.SVC(kernel='poly')),('svmrbf', svm.SVC(kernel='rbf'))],
			voting='hard'
		  )


		self.num_classes = num_classes


	def train(self,features,labels):
		"""
        Using a set of features and labels, trains the classifier and returns the training accuracy.
        :param features: An MxN matrix of features to use in prediction
        :param labels: An M row list of labels to train to predict
        :return: Prediction accuracy, as a float between 0 and 1
        """

		labels = self.labels_to_categorical(labels)
		self.model.fit(features,labels)
		accuracy = self.model.score(features,labels)
		return accuracy

	# make sure you save model using the same library as we used in machine learning price-predictor

	def predict(self,features,labels):
		"""
        Using a set of features and labels, predicts the labels from the features,
        and returns the accuracy of predicted vs actual labels.
        :param features: An MxN matrix of features to use in prediction
        :param labels: An M row list of labels to test prediction accuracy on
        :return: Prediction accuracy, as a float between 0 and 1
        """
		label_train = self.labels_to_categorical(labels)
		labels = self.model.predict(features)
		accuracy = self.model.score(features,label_train)
		return accuracy

	def get_prediction(self,features):
		return self.model.predict(features)

	def reset(self):
		"""
        Resets the trained weights / parameters to initial state
        :return:
        """

		pass

	def labels_to_categorical(self,labels):
		_,IDs = unique(labels,return_inverse=True)
		return IDs
