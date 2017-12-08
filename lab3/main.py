import matplotlib.pyplot as plt
import pandas as pnd
from scipy.stats import pearsonr
from data import Data


csv = pnd.read_csv('../data/iris.csv')
data = Data(csv, 0.9)

plt.figure(figsize=(8, 6))

for label,marker,color in zip(
        ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),('o', 'x', '^'),('red', 'blue', 'green')):

    R = pearsonr(data.train_x[:,0][data.train_y == label], data.train_x[:,3][data.train_y == label])
    plt.scatter(x=data.train_x[:,0][data.train_y == label],
                y=data.train_x[:,3][data.train_y == label],
                marker=marker,
                color=color,
                alpha=0.7,
                label='class {:}, R={:.2f}'.format(label, R[0])
                )

plt.title('Iris Dataset')
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.legend(loc='lower right')

plt.show()