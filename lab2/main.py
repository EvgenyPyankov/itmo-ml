import pandas as pnd
from lab2.experiment import Experiment

data = pnd.read_csv('../data/iris.csv')

open('docs/results.txt', 'w').close()

exp1 = Experiment(1, 0.6, data)
exp2 = Experiment(2, 0.7, data)
exp3 = Experiment(3, 0.8, data)
exp4 = Experiment(4, 0.9, data)
exp1.run()
exp2.run()
exp3.run()
exp4.run()
