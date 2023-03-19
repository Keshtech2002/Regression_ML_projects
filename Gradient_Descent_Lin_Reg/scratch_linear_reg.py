import numpy as np


class Linear_Regression:
    # Initiating the parameters (learning_rate & n0_of_iterations)
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations


    def fit(self, X, Y):
        # number of training example and number of features
        self.m, self.n = X.shape # number of rows & columns

        # initiating the weight and bias
        self.w = np.zeros(self.n) # this is the estimator for each of the features, for every feature there is a slope
                                    # that is why we have a numpy array to contain all the feature estimators
        self.b = 0 # intercept
        self.X = X
        self.Y = Y

        # implementing Gradient Descent
        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        Y_prediction = self.predict(self.X)

        # calculate gradient
        dw = - (2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
        db = - 2 * np.sum(self.Y - Y_prediction) / self.m

        # update the weigths
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        return X.dot(self.w) + self.b
