import pandas as pnd
from data import Data
from lab2.experiment import Experiment

data = pnd.read_csv('../data/iris.csv')

open('docs/results.txt', 'w').close()

exp1 = Experiment(1, Data(data, 0.6))
exp2 = Experiment(2, Data(data, 0.7))
exp3 = Experiment(3, Data(data, 0.8))
exp4 = Experiment(4, Data(data, 0.9))
exp1.run()
exp2.run()
exp3.run()
exp4.run()
