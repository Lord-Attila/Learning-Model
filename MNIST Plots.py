import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#print("x_train shape:", x_train.shape)
#print("y_train shape:", y_train.shape)
#print("x_test shape:", x_test.shape)
#print("y_test shape:", y_test.shape)

i = 1
plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
plt.show()
print(y_train[i])