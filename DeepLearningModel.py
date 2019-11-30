import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class DeepLearningModel(object):
    def __init__(self,inputsize):
        self.inputsize = inputsize

    def build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(512,activation='relu',input_shape=(self.inputsize,)))
        model.add(keras.layers.Dense(512,activation='relu'))
        model.add(keras.layers.Dense(256,activation='relu'))
        model.add(keras.layers.Dense(256,activation='relu'))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(1))
        model.compile(optimizer='adam', loss=keras.losses.mse, metrics=['mae'])
        return model

    def k_fold_vaildation(self,x_train,y_train,K,epoches):
        num_val_samples = len(x_train)//K
        all_rmse_histories = []
        all_val_rmse_histories = []
        all_mae_histories = []
        for i in range(K):
            print('processing fold #',i)
            val_data = x_train[i*num_val_samples:(i+1)*num_val_samples]
            val_targets =y_train[i*num_val_samples:(i+1)*num_val_samples]
            partial_train_data = np.concatenate([x_train[:i*num_val_samples],x_train[(i+1)*num_val_samples:]],axis=0)
            partial_train_targets = np.concatenate([y_train[:i*num_val_samples],y_train[(i+1)*num_val_samples:]],axis=0)
            model = self.build_model()
            history = model.fit(partial_train_data,partial_train_targets,validation_data=(val_data,val_targets),epochs=epoches)
            rmse_history = history.history['loss']
            val_rmse_history = history.history['val_loss']
            mae_history = history.history['val_mean_absolute_error']
            all_rmse_histories.append(rmse_history)
            all_val_rmse_histories.append(val_rmse_history)
            all_mae_histories.append(mae_history)

        average_rmea_history = [
            np.sqrt(np.mean([x[i] for x in all_rmse_histories])) for i in range(epoches)
        ]
        average_val_rmea_history = [
            np.sqrt(np.mean([x[i] for x in all_val_rmse_histories])) for i in range(epoches)
        ]
        average_mae_history = [
            np.mean([x[i] for x in all_mae_histories]) for i in range(epoches)
        ]
        plt.figure(1)
        plt.plot(range(1,len(average_rmea_history[30:])+1),average_rmea_history[30:],'bo',label='Training loss')
        plt.plot(range(1, len(average_val_rmea_history[30:]) + 1), average_val_rmea_history[30:], 'b', label='Validation loss')
        plt.legend()
        plt.show()

        smooth_rmse_history = self.smooth_curve(average_rmea_history[10:])
        smooth_val_rmse_history = self.smooth_curve(average_val_rmea_history[10:])
        smooth_mae_history = self.smooth_curve(average_mae_history[10:])
        plt.figure(2)
        plt.plot(range(1, len(smooth_rmse_history) + 1), smooth_rmse_history , 'ro', label='Training loss')
        plt.plot(range(1, len(smooth_val_rmse_history) + 1), smooth_val_rmse_history, 'r', label='Validation loss')
        plt.legend()
        plt.show()
        plt.figure(3)
        plt.plot(range(1,len(smooth_mae_history)+1), smooth_mae_history)
        plt.show()

    def smooth_curve(self,points,factor=0.9):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous*factor+point*(1-factor))
            else:
                smoothed_points.append(point)
        return smoothed_points
