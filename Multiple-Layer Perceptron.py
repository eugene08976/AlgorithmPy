import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
np.set_printoptions(precision=2)
class MLP():
    def __init__(self, listOfLayer=[2, 3, 2]):
        self.numLayers = len(listOfLayer)
        self.layers = [{} for i in range(self.numLayers)]
        
        for i in range(1, self.numLayers):
            self.layers[i]["W"] = (np.random.randn(listOfLayer[i], listOfLayer[i-1]) - 0.5) / 10000.0
            self.layers[i]["b"] = (np.random.randn(listOfLayer[i], 1) - 0.5 / 10000.0)
        self.lossList = []
        return
            
    def forward(self, x, bs):
        self.layers[0]["a"] = np.copy(x)
        for i in range(1, self.numLayers):
            self.layers[i]["z"] = self.layers[i]["W"].dot(self.layers[i-1]["a"]) + self.layers[i]["b"]
            self.layers[i]["a"] = self.activation(self.layers[i]["z"])
        self.p = self.softmax(self.layers[-1]["a"])
        self.yHat = np.zeros(self.p.shape, dtype=int)
        for j, i in enumerate(self.p.argmax(axis=0)):
            self.yHat[i, j] = 1
        return
    
    def activation(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def inverseActivation(self, a):
        return a * (1.0 - a)
    
    def softmax(self, a):
        expa = np.exp(a)
        return (expa / np.sum(expa, axis = 0))
    
    def loss(self, y):
        return np.mean(- np.log(self.p) * y - np.log(1.0 - self.p) * (1.0 - y))
        
    def backprop(self, y, bs):
        self.dJdp = - y / self.p + (1.0 - y) / (1.0 - self.p)
        dpda = np.zeros((self.p.shape[0], self.p.shape[0], bs))
        for b in range(bs):
            for i in range(dpda.shape[0]):
                for j in range(dpda.shape[1]):
                    dpda[i, j, b] = (self.p[i, b] - self.p[i, b] ** 2) if i == j else - self.p[i, b] * self.p[j, b]
        
        self.layers[-1]["dJda"] = np.zeros(self.layers[-1]["a"].shape)
        for b in range(bs):
            self.layers[-1]["dJda"][:, b] = dpda[:, :, b].dot(self.dJdp[:, b])
          
        for i in range(self.numLayers - 1, 0, -1):
            self.layers[i]["dJdz"] = self.inverseActivation(self.layers[i]["a"]) * self.layers[i]["dJda"]
            self.layers[i]["dJdb"] = np.mean(self.layers[i]["dJdz"], axis = 1).reshape(self.layers[i]["b"].shape)
            self.layers[i]["dJdW"] = self.layers[i]["dJdz"].dot(self.layers[i-1]["a"].T) / bs
            self.layers[i-1]["dJda"] = self.layers[i]["W"].T.dot(self.layers[i]["dJdz"])
        return
            
    def update(self, lr = 0.01):
        for i in range(1, self.numLayers):
            self.layers[i]["W"] -= lr * self.layers[i]["dJdW"]
            self.layers[i]["b"] -= lr * self.layers[i]["dJdb"]
        return
    
    def shuffle(self, a, b):
        shuffled_a = np.copy(a)
        shuffled_b = np.copy(b)
        permutation = np.random.permutation(a.shape[1])
        for oldindex, newindex in enumerate(permutation):
            shuffled_a[:, oldindex] = a[:, newindex]
            shuffled_b[:, oldindex] = b[:, newindex]
        return shuffled_a, shuffled_b
        
        
    def train(self, trainX, trainY, numEpoch=1, lr=0.01, bs=2):  
        for e in range(numEpoch):
            shuffled_trainX, shuffled_trainY = self.shuffle(trainX, trainY)
            for i in range(trainX.shape[1] // bs):
                x = shuffled_trainX[:, i*bs : (i+1)*bs]
                y = shuffled_trainY[:, i*bs : (i+1)*bs]
                self.forward(x, bs)
                self.lossList.append(self.loss(y))
                self.backprop(y, bs)
                self.update(lr)
        return      
        
nn = MLP([2, 5, 2])
trainX = np.array([[1, 4, 2, 1, 2, 4],
                  [1, 2, 2, 4, 3, 4]])
trainY = np.array([[0, 0, 0, 1, 1, 0],
                  [1, 1, 1, 0, 0, 1]])
nn.train(trainX, trainY, numEpoch=10000, lr=0.1, bs = 6)
plt.plot(nn.lossList[:])

print ("yHat is\n", nn.yHat)
print ("y is \n", trainY)
nn.p

f = open("mnist_test_10.csv", "r")
a = f.readlines()
f.close()

x = []
y = []
count = 1
for line in a:
    linepixels = [int(pixel) for pixel in line.split(",")]
    x.append(linepixels[1:])
    y.append(linepixels[0])
    imarray = np.asfarray(linepixels[1:]).reshape(28,28)
    plt.subplot(5, 5, count)
    plt.subplots_adjust(hspace=0.5)
    plt.title("Label is" + str(linepixels[0]))
    count += 1
    plt.imshow(imarray, cmap="Greys", interpolation="None")
    
test_x_10 = np.clip(np.array(x).T, 0, 1)
test_y_10 = np.zeros((10, len(y)), dtype=int)
for i in range(len(y)):
    test_y_10[y[i], i] = 1
    
test_x_10.shape
test_y_10
nnMNIST = MLP([784, 50, 10])
nnMNIST.train(test_x_10, test_y_10, numEpoch=10000, lr=0.1, bs = 10)
plt.plot(nnMNIST.lossList[:])
nnMNIST.yHat.argmax(axis=0)
f = open("mnist_train_100.csv", "r")
a = f.readlines()
f.close()

x = []
y = []
count = 1
for line in a:
    linepixels = [int(pixel) for pixel in line.split(",")]
    x.append(linepixels[1:])
    y.append(linepixels[0])
    imarray = np.asfarray(linepixels[1:]).reshape(28,28)
    plt.subplot(10, 10, count)
    plt.subplots_adjust(hspace=0.5)
    plt.title("Label is" + str(linepixels[0]))
    count += 1
    plt.imshow(imarray, cmap="Greys", interpolation="None")
    
train_x_100 = np.clip(np.array(x).T, 0, 1)
train_y_100 = np.zeros((10, len(y)), dtype=int)
for i in range(len(y)):
    train_y_100[y[i], i] = 1

nnMNIST100 = MLP([784, 32, 10])
nnMNIST100.train(train_x_100, train_y_100, numEpoch=300, lr=0.1, bs = 10)
plt.plot(nnMNIST100.lossList[:])

def csv2xy(fileName):
    f = open(fileName, "r")
    a = f.readlines()
    f.close()

    x = []
    y = []

    for line in a:
        linepixels = [int(pixel) for pixel in line.split(",")]
        x.append(linepixels[1:])
        y.append(linepixels[0])
    
    
    out_x = np.clip(np.array(x).T, 0, 1)
    out_y = np.zeros((10, len(y)), dtype=int)
    for i in range(len(y)):
        out_y[y[i], i] = 1
        
    return out_x, out_y
    
train_x_60000, train_y_60000 = csv2xy("mnist_train_60000.csv")
test_x_10000, test_y_10000 = csv2xy("mnist_test_10000.csv")
