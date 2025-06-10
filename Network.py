import numpy as np
import matplotlib.pyplot as plt

class network:
    def __init__(self, neurons, output_size, alpha, X_train):
        self.w2 = None
        self.neurons = neurons
        self.output_size = output_size
        self.alpha = alpha
        size = X_train.shape[0]
        self.w1 = np.random.rand(self.neurons, size) - 0.5
        self.w2 = np.random.rand(self.neurons, self.neurons) - 0.5
        self.w3 = np.random.rand(self.output_size, self.neurons) - 0.5
        self.b1 = np.random.rand(self.neurons, 1) - 0.5
        self.b2 = np.random.rand(self.neurons, 1) - 0.5
        self.b3 = np.random.rand(self.output_size, 1) - 0.5

    def ReLU(self, a):
        return np.maximum(a, 0)

    def ReLU_derivative(self, a):
        return a > 0

    def softmax(self, a):
        return np.exp(a)/sum(np.exp(a))

    def sigmoid(self, a):
        return 1/(1+np.exp(-a))

    def sigmoid_derivative(self, a):
        return self.sigmoid(a)*(1-self.sigmoid(a))

    def one_hot(self, a):
        ones_array = np.zeros((a.size, a.max()+1))
        for i in range(a.size):
            ones_array[i,a[i]] = 1
        ones_array = ones_array.T
        return ones_array

    def prediction(self, a):
        return np.argmax(a, 0)

    def accuracy(self, a, b):
        return np.sum(a == b) / b.size

    def matching(self, a, b):
        return np.sum(a == b)

    def import_parameters(self, w1, w2, w3, b1, b2, b3):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def accuracy_graph(self, a, epoch):
        b = np.linspace(0, epoch, epoch)
        fs = 20
        plt.plot(b, a)
        plt.title('Network Accuracy vs Epoch')
        plt.xlabel('Epoch',{'fontsize':fs})
        plt.ylabel('Accuracy',{'fontsize':fs})
        plt.ylim([0, 100])
        plt.show()

    def export_parameters(self, epoch):
        np.savetxt("w1("+str(epoch)+".txt)", self.w1, delimiter=",", fmt='%d')
        np.savetxt("w2"+str(epoch)+".txt)", self.w2, delimiter=",", fmt='%d')
        np.savetxt("w3"+str(epoch)+".txt)", self.w3, delimiter=",", fmt='%d')
        np.savetxt("b1"+str(epoch)+".txt)", self.b1, delimiter=",", fmt='%d')
        np.savetxt("b2"+str(epoch)+".txt)", self.b2, delimiter=",", fmt='%d')
        np.savetxt("b3"+str(epoch)+".txt)", self.b3, delimiter=",", fmt='%d')

    def forward(self, X_train):
        z1 = np.matmul(self.w1, X_train) + self.b1
        a1 = self.ReLU(z1)
        z2 = np.matmul(self.w2, a1) + self.b2
        a2 = self.ReLU(z2)
        z3 = np.matmul(self.w3, a2) + self.b3
        a3 = self.softmax(z3)
        return z1, z2, z3, a1, a2, a3

    def backward(self, m, z1, z2, a1, a2, a3, X, Y):
        delta_3  = a3 - self.one_hot(Y)
        delta_2 = self.w3.T.dot(delta_3) * self.ReLU_derivative(z2)
        delta_1 = self.w2.T.dot(delta_2)*self.ReLU_derivative(z1)
        d_w3 = 1/m*delta_3.dot(a2.T)
        d_w2 = 1/m*delta_2.dot(a1.T)
        d_w1 = 1/m*delta_1.dot(X.T)
        return d_w1, d_w2, d_w3, delta_1, delta_2, delta_3

    def update(self, m, d_w1, d_w2, d_w3, delta_1, delta_2, delta_3):
        self.w1 = self.w1 - self.alpha * d_w1
        self.w2 = self.w2 - self.alpha * d_w2
        self.w3 = self.w3 - self.alpha * d_w3
        self.b1 = self.b1 - self.alpha * 1/m * np.sum(delta_1)
        self.b2 = self.b2 - self.alpha * 1/m * np.sum(delta_2)
        self.b3 = self.b3 - self.alpha * 1/m * np.sum(delta_3)

    def training(self, epoch, X_train, Y_train):
        m = X_train.shape[1]
#        gradient_array  = []
        accuracy_array = []
        for i in range(epoch):
            z1, z2, z3, a1, a2, a3 = self.forward(X_train)
            d_w1, d_w2, d_w3, delta_1, delta_2, delta_3 = self.backward(m, z1, z2, a1, a2, a3, X_train, Y_train)
            self.update(m, d_w1, d_w2, d_w3, delta_1, delta_2, delta_3)
            accuracy_array.append((self.accuracy(self.prediction(a3), Y_train) * 100))
            if i % 10 == 0:
                print("Iteration:", i+10)
                print("Prediction:", self.prediction(a3))
                print("Accuracy:", self.accuracy(self.prediction(a3), Y_train) * 100, "%")
        return accuracy_array

    def testing(self, X_test, Y_test):
        z1, z2, z3, a1, a2, a3 = self.forward(X_test)
        print("TESTING")
        print("Predictions: ", self.prediction(a3))
        print("Expected: ", Y_test)
        print("Accuracy: ", self.accuracy(self.prediction(a3), Y_test) * 100, "%")
        print("Correct Estimates: ", self.matching(self.prediction(a3), Y_test))