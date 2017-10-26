import pandas as pnd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from utils import print_results
from data import Data
from lab1.kns_classifier import KNsClassifier
from lab1.naive_bayes_classifier import NBClassifier

csv = pnd.read_csv('../data/iris.csv')
data = Data(csv, 0.7)

sklearn_nb_clf = GaussianNB()
sklearn_nb_clf.fit(data.train_x, data.train_y)
sklearn_nb_clf_accuracy = accuracy_score(data.test_y, sklearn_nb_clf.predict(data.test_x))

my_nb_clf = NBClassifier()
my_nb_clf.fit(data.train_x, data.train_y)
my_nb_clf_accuracy = accuracy_score(data.test_y, my_nb_clf.predict(data.test_x))

k = 10

my_kn_clf = KNsClassifier(k)
my_kn_clf.fit(data.train_x, data.train_y)
my_kn_clf_accuracy = accuracy_score(data.test_y, my_kn_clf.predict(data.test_x))

sklearn_kn_clf = KNeighborsClassifier(k)
sklearn_kn_clf.fit(data.train_x, data.train_y)
sklearn_kn_clf_accuracy = accuracy_score(data.test_y, sklearn_kn_clf.predict(data.test_x))

text = 'Lab #1\n'\
       'Sklearn Naive Bayes classifier accuracy = {:0.4f}%\n'\
    'My Naive Bayes classifier accuracy = {:0.4f}%\n\n'\
    'Sklearn K Nearest Neighbours classifier accuracy = {:0.4f}%\n'\
    'My K Nearest Neighbours classifier accuracy = {:0.4f}%'.format(sklearn_nb_clf_accuracy,
                                                                    my_nb_clf_accuracy,
                                                                    sklearn_kn_clf_accuracy,
                                                                    my_kn_clf_accuracy)

print_results(text)
