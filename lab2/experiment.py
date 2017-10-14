from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import print_results


class Experiment:
    def __init__(self, num, train_percentage, data):
        assert train_percentage < 1
        self.num = num
        self.train_percentage = train_percentage
        self.train, self.test = train_test_split(data, train_size=train_percentage)
        self.dtc = DecisionTreeClassifier()
        self.rfc = RandomForestClassifier()

    def print_results(self, dtc_accuracy, rfc_accuracy):
        text = 'Experiment #{}\n' \
               'Training percentage = {:0.0f}%\n' \
               'Dicision Tree Classifier accuracy = {:0.2f}%\n' \
               'Randrom Forest Classifier accuracy = {:0.2f}%\n\n'.format(self.num,
                                                                          self.train_percentage * 100,
                                                                          dtc_accuracy * 100,
                                                                          rfc_accuracy * 100)
        print_results(text)

    def run(self):
        x_attributes = ['Slength', 'Swidth', 'Plength', 'Pwidth']
        X = self.train.loc[:, x_attributes]
        Y = self.train['Class']
        self.dtc.fit(X, Y)
        self.rfc.fit(X, Y)
        dtc_predict = self.dtc.predict(self.test.loc[:, x_attributes])
        rfc_predict = self.rfc.predict(self.test.loc[:, x_attributes])
        dtc_accuracy = accuracy_score(self.test['Class'], dtc_predict)
        rfc_accuracy = accuracy_score(self.test['Class'], rfc_predict)
        self.print_results(dtc_accuracy, rfc_accuracy);
