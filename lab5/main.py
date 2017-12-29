import numpy as np
import pandas as pnd
from sklearn import cross_validation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

from data import Data

csv = pnd.read_csv('../data/banknote.csv')
kFold = cross_validation.KFold(n=len(csv), random_state=50, shuffle=True)
gnb = GaussianNB()
lda = LDA()


def classification_accuracy(X, y):
    gnb = GaussianNB()
    scores = cross_val_score(gnb, X, y, cv=kFold, scoring='accuracy')
    print("Accuracy for GaussianNB: %0.3f (%0.3f)" % (scores.mean(), scores.std()))

    lda = LDA()
    scores = cross_val_score(lda, X, y, cv=kFold, scoring='accuracy')
    print("Accuracy for LDA: %0.3f (%0.3f)" % (scores.mean(), scores.std()))


def logarithmic_loss(X, y):
    scores = cross_val_score(gnb, X, y, cv=kFold, scoring='neg_log_loss')
    print("Logarithmic loss for GaussianNB: %0.3f (%0.3f)" % (scores.mean(), scores.std()))

    scores = cross_val_score(lda, X, y, cv=kFold, scoring='neg_log_loss')
    print("Logarithmic loss for LDA: %0.3f (%0.3f)" % (scores.mean(), scores.std()))


def area_under_roc_curve(X, y):
    scores = cross_val_score(gnb, X, y, cv=kFold, scoring='roc_auc')
    print("Area under ROC curve for GaussianNB: %0.3f (%0.3f)" % (scores.mean(), scores.std()))

    scores = cross_val_score(lda, X, y, cv=kFold, scoring='roc_auc')
    print("Area under ROC curve for LDA: %0.3f (%0.3f)" % (scores.mean(), scores.std()))


def confusion_matrix(test_y, gnb_predict, lda_predict):
    from sklearn.metrics import confusion_matrix

    gnb_matrix = confusion_matrix(test_y, gnb_predict)
    print("Confusion matrix for GaussianNB")
    print(gnb_matrix)

    lda_matrix = confusion_matrix(test_y, lda_predict)
    print("Confusion matrix for LDA")
    print(lda_matrix)


def classification_report(test_y, gnb_predict, lda_predict):
    from sklearn.metrics import classification_report
    gnb_report = classification_report(test_y, gnb_predict)
    print('Classification report for GaussianNB:')
    print(gnb_report)

    lda_report = classification_report(test_y, lda_predict)
    print('Classification report for LDA:')
    print(lda_report)


def main():
    data = Data(csv, 0.7)

    X = csv.values[:, 0: -1]
    y = (csv.values[:, -1]).astype(np.int, copy=False)

    gnb.fit(data.train_x, data.train_y)
    gnb_predict = gnb.predict(data.test_x)

    lda.fit(data.train_x, data.train_y)
    lda_predict = lda.predict(data.test_x)

    classification_accuracy(X, y)
    logarithmic_loss(X, y)
    area_under_roc_curve(X, y)
    confusion_matrix(data.test_y, gnb_predict, lda_predict)
    classification_report(data.test_y, gnb_predict, lda_predict)


main()
