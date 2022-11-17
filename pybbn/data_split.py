from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import numpy

data = genfromtxt('output2.csv', delimiter=',')

x_train, x_test, = train_test_split(data,test_size=0.5)

print(x_train)