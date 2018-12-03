from sklearn.utils import resample, shuffle
import numpy as np

def resample_training_data(x_train, y_train):
    x_train_0 = []
    x_train_1 = []
    y_train_0 = []
    y_train_1 = []

    for i in range(len(x_train)):
        if y_train[i] == 0:
            x_train_0.append(x_train[i])
            y_train_0.append(y_train[i])

        else:
            x_train_1.append(x_train[i])
            y_train_1.append(y_train[i])

    x_train_1, y_train_1 = resample(x_train_1, y_train_1, n_samples=len(x_train_0), replace=True)

    x_train_new = np.array(shuffle(x_train_0+x_train_1))
    y_train_new = np.array(shuffle(y_train_0+y_train_1))

    return x_train_new, y_train_new
