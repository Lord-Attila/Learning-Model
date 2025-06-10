from numpy import genfromtxt
import tensorflow as tf
from Network import *

#(x_train, Y_train), (x_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()

#genfromtxt('my_file.csv', delimiter=',')
name = '(100000)'
w1 = np.loadtxt('w1' + name + '.txt', delimiter=',')
w2 = np.loadtxt('w2' + name + '.txt', delimiter=',')
w3 = np.loadtxt('w3' + name + '.txt', delimiter=',')
b1 = np.loadtxt('b1' + name + '.txt', delimiter=',')
b2 = np.loadtxt('b2' + name + '.txt', delimiter=',')
b3 = np.loadtxt('b3' + name + '.txt', delimiter=',')

(x_train, Y_train), (x_test, Y_test) = tf.keras.datasets.mnist.load_data()
X_train = x_train.reshape(60000,784).T
X_train = X_train/255
X_test = x_test.reshape(10000, 784).T
X_test = X_test/255

epoch = 10
nn = network(50, 10, 0.1, X_train)
accuracy_array = nn.training(epoch, X_train, Y_train)
nn.testing(X_test, Y_test)
nn.import_parameters(w1, w2, w3, b1, b2, b3)
nn.testing(X_test, Y_test)
nn.accuracy_graph(accuracy_array, epoch)