import pandas as pnd
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import prepare_file

# sys.path.append('..')
data = pnd.read_csv('../data/iris.csv')
print(data)
x_attributes = ['Slength', 'Swidth', 'Plength', 'Pwidth'];
X = data.loc[:, x_attributes]
Y = data['Class']
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
dtc.fit(X,Y)
arr = [[7.2, 3.0, 5.8, 1.6 ]]
print(dtc.predict(arr))
# arr = [[5.1], [3.5], [1.4], [0.2]]

