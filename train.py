from random import seed

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np

from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import KBinsDiscretizer

import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

seed(41)

curve = 'roc'
balance_data = False

est = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')

x_train = np.load('./x_train.npy')
y_train_raw = np.load('./y_train.npy')
y_train = np.array([0 if x < 2.0 else 1 for x in y_train_raw])

x_test = np.load('./x_test.npy')
y_test_raw = np.load('./y_test.npy')
y_test = np.array([0 if x < 2.0 else 1 for x in y_test_raw])




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
    plt.title("Raining Classifier ROC")

    plt.show()

elif curve == 'pr':
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)

    plt.plot(precision, recall)

    plt.plot([1, 0], [0, 1], 'k--')

    plt.legend(["rain classifier", "randomly guessing"])

    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Raining Classifier Precision/Recall")

    plt.show()






