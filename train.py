from random import seed

import numpy as np

from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.utils import resample

import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

from util import resample_training_data

seed(42)

curve = 'roc'
balance_data = False

est = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')


x_train = np.load('./x_train.npy')
y_train_raw = np.load('./y_train.npy')
y_train = np.array([0 if x < 2.0 else 1 for x in y_train_raw])

if balance_data:
    x_train, y_train = resample_training_data(x_train, y_train)



x_test = np.load('./x_test.npy')
y_test_raw = np.load('./y_test.npy')
y_test = np.array([0 if x < 2.0 else 1 for x in y_test_raw])

if balance_data:
    x_test, y_test = resample_training_data(x_test, y_test)


clf = LinearSVC()

clf.fit(x_train, y_train)


y_pred = clf.predict(x_test)
y_score = clf.decision_function(x_test)

y_zeros = np.zeros((len(y_pred)))

print("predicted accuracy: ", accuracy_score(y_test, y_pred))
print("zeros accuracy: ", accuracy_score(y_test, y_zeros))

if curve == 'roc':
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)

    plt.plot(fpr, tpr)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.legend(["rain classifier (area = %0.2f)" % roc_auc, "randomly guessing"])

    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    if balance_data:
        plt.title("Raining Classifier Balanced ROC")
    else:
        plt.title("Raining Classifier ROC")

    plt.show()

elif curve == 'pr':
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)

    plt.plot(recall, precision)

    plt.legend(["rain classifier (average precision = %0.2f)" % average_precision])

    plt.ylabel("Precision")
    plt.xlabel("Recall")

    if balance_data:
        plt.title("Raining Classifier Balanced Precision/Recall")
    else:
        plt.title("Raining Classifier Precision/Recall")

    plt.show()






