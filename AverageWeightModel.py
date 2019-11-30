#加权模型
from sklearn.base import BaseEstimator, RegressorMixin, clone
import numpy as np

class AverageWeightModel(BaseEstimator, RegressorMixin):
    def __init__(self,method,weight):
        self.method = method
        self.weight = weight

    def fit(self,x,y):
        self.models_ = [clone(x) for x in self.method]
        for model in self.models_:
            model.fit(x,y)
        return self

    def predict(self,x):
        w = list()
        pred = np.array([model.predict(x) for model in self.models_])
        for data in range(pred.shape[1]):
            single = [pred[model,data]*weight for model,weight in zip(range(pred.shape[0]), self.weight)]
            w.append(np.sum(single))
        return w