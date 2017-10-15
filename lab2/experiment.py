from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from utils import print_results


class Experiment:
    def __init__(self, num, data):
        self.num = num
        self.data = data
        self.dtc = DecisionTreeClassifier()
        self.rfc = RandomForestClassifier()

    def print_results(self, dtc_accuracy, rfc_accuracy):
        text = 'Experiment #{}\n' \
               'Training percentage = {:0.0f}%\n' \
               'Dicision Tree Classifier accuracy = {:0.2f}%\n' \
               'Randrom Forest Classifier accuracy = {:0.2f}%\n\n'.format(self.num,
                                                                          self.data.train_size * 100,
                                                                          dtc_accuracy * 100,
                                                                          rfc_accuracy * 100)
        print_results(text)

    def run(self):
        self.dtc.fit(self.data.train_x, self.data.train_y)
        self.rfc.fit(self.data.train_x, self.data.train_y)
        dtc_predict = self.dtc.predict(self.data.test_x)
        rfc_predict = self.rfc.predict(self.data.test_x)
        dtc_accuracy = accuracy_score(self.data.test_y, dtc_predict)
        rfc_accuracy = accuracy_score(self.data.test_y, rfc_predict)
        self.print_results(dtc_accuracy, rfc_accuracy);
