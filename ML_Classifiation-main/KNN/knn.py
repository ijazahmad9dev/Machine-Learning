# This Code mainly focus on the implementation of KNN.
# This code will help to understand the KNN implementation
# from scrah.

# Author: Ijaz Ahmad

import numpy as np
from collections import Counter

def dist(x1, x2):
    dis = np.sqrt(np.sum((x1 - x2)**2))
    return dis

class KNN():
    def __init__(self, k = 3):
        self.k = k
    def fit(self, X_train, y_train):
        self.X_trains = X_train
        self.y_trains = y_train
    def predic(self, X_test):
        predictions = []
        for x in X_test:
            pred = self._predic(x)
            predictions.append(pred)
        
        return predictions
    def _predic(self, x):
        distance = []
        for x_train in self.X_trains:
            dis = dist(x, x_train)
            distance.append(dis)
        indices = np.argsort(distance)[:self.k]
        labels = []
        for i in indices:
            indx = self.y_trains[i]
            labels.append(indx)
        
        return Counter(labels).most_common()[0][0]



