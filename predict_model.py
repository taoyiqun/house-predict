from sklearn.base import BaseEstimator, RegressorMixin, clone
from data_preprocessing import data_preprocess
from data_preprocessing import show_data
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from AverageWeightModel import AverageWeightModel
from DeepLearningModel import DeepLearningModel
import keras
#导入处理好的数据
def load_data():
    train_df, test_df, sale_price, oridatas = data_preprocess()
    x_train = train_df.values
    x_test = test_df.values
    y_train = np.array(sale_price)
    return x_train, y_train, x_test, oridatas

#计算均方误差
def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
    return rmse

#岭回归调参
def Ridge_Model():
    x_train, y_train, x_test,_ = load_data()
    alphas = np.logspace(-3,2,50)
    test_scores = []
    i = 0
    for alpha in alphas:
        i += 1
        print(i)
        clf = Ridge(alpha)
        test_score = np.sqrt(-cross_val_score(clf,x_train,y_train,cv=10,scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))
    print('最佳参数为：')
    print(alphas[test_scores.index(min(test_scores))])
    print("最优结果：")
    print(min(test_scores))
    figure = plt.figure(0)
    plt.plot(alphas,test_scores)
    plt.title('Alpha vs CV Error')
    plt.xlabel('alpha')
    plt.ylabel('CV Error')
    plt.savefig('岭回归调参')
    plt.close(0)

#袋装法调参
def BaggingRegressor_Model():
    #best 30
    best_alpha_of_ridge = 9.540
    x_train, y_train, x_test,_ = load_data()
    params = [1,10,15,20,25,30,40]
    test_scores = []
    for param in params:
        print(param)
        clf = BaggingRegressor(base_estimator=Ridge(best_alpha_of_ridge),n_estimators=param)
        test_score = np.sqrt(-cross_val_score(clf, x_train, y_train, cv=10, scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))
    print('最佳参数为：')
    print(params[test_scores.index(min(test_scores))])
    print("最优结果：")
    print(min(test_scores))
    figure = plt.figure(0)
    plt.plot(params, test_scores)
    plt.title('Params vs CV Error')
    plt.xlabel('params')
    plt.ylabel('CV Error')
    plt.savefig('袋装法调参')
    plt.close(0)


#贝叶斯回归
def BayesianRidge_Model():
    x_train, y_train, x_test,_ = load_data()
    clf = BayesianRidge()
    test_score = np.sqrt(-cross_val_score(clf, x_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    print(np.mean(test_score))

#核函数岭回归调参
def KernelRidge_Model():
    #best 9.3
    x_train, y_train, x_test,_ = load_data()
    alphas = np.arange(9,9.7,0.1)
    test_scores = []
    for alpha in alphas:
        print(alpha)
        clf = KernelRidge(alpha=alpha,kernel='linear')
        test_score = np.sqrt(-cross_val_score(clf,x_train,y_train,cv=10,scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))
    print('最佳参数为：')
    print(alphas[test_scores.index(min(test_scores))])
    print("最优结果：")
    print(min(test_scores))
    figure = plt.figure(0)
    plt.plot(alphas,test_scores)
    plt.title('Alpha vs CV Error')
    plt.xlabel('alpha')
    plt.ylabel('CV Error')
    plt.savefig('核函数岭回归调参')
    plt.close(0)


#XGB增强树调参
def XGBRegressor_Model():
    x_train, y_train, x_test, _ = load_data()
    params = np.arange(1,10,1)
    test_scores = []
    for param in params:
        print(param)
        clf = XGBRegressor(max_depth=param,learning_rate=0.1)
        test_score = np.sqrt(-cross_val_score(clf, x_train, y_train, cv=10, scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))
    print('最佳参数为：')
    print(params[test_scores.index(min(test_scores))])
    print("最优结果：")
    print(min(test_scores))
    figure = plt.figure(0)
    plt.plot(params, test_scores)
    plt.title('Max_depth vs CV Error')
    plt.xlabel('max_depth')
    plt.ylabel('CV Error')
    plt.savefig('XGB增强树调参')
    plt.close(0)


def Ave_Model():
    ridge = Ridge(alpha=9.540)
    baggingregressor = BaggingRegressor(base_estimator=ridge,n_estimators=30)
    bayesianridge = BayesianRidge()
    kernelridge = KernelRidge(alpha=9.3,kernel='linear')
    xgbregressor = XGBRegressor(max_depth=5,learning_rate=0.1)
    method = [ridge, baggingregressor, bayesianridge, kernelridge, xgbregressor]
    weight_avg = AverageWeightModel(method = method, weight=[0.2]*5)
    return weight_avg

def PredictByAllModel():
    train_df, test_df, sale_price, _ = data_preprocess()
    x_train, y_train, x_test, oridata = load_data()
    ave_model = Ave_Model()
    ave_model.fit(x_train, y_train)
    y_predict1 = np.array(ave_model.predict(x_test))
    deep_learning_model = DeepLearningModel(x_train.shape[1])
    deeplearningModel = deep_learning_model.build_model()
    deeplearningModel.fit(x_train,y_train,epochs=300)
    deeplearningModel.save('model.h5')
    prices = []
    for i in range(x_test.shape[0]):
        print(i)
        price = x_test[i].reshape((1, x_test.shape[1]))
        prices.append(deeplearningModel.predict(price))
        i += 1
    y_predict2 = np.array(prices).reshape((-1,))
    w1 = 0.9
    w2 = 0.1
    y_predict = y_predict1 * w1 + y_predict2 * w2
    y_predict *= (oridata['SalePrice']['max']-oridata['SalePrice']['min'])
    y_predict += oridata['SalePrice']['min']
    y_predict = np.expm1(y_predict)
    submission_df = pd.DataFrame(data={'Id':test_df.index,'SalePrice':y_predict})
    print(submission_df.head(10))
    submission_df.to_csv('submission11.csv',index=False)

if __name__ == '__main__':
#    show_data()
#    Ridge_Model()
#    BaggingRegressor_Model()
#    KernelRidge_Model()
#    XGBRegressor_Model()
    PredictByAllModel()