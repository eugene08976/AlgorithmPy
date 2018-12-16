import numpy as np
import math
%matplotlib inline
import matplotlib.pylab as plt
class Conv_Layer(object):
    '''Convolution Layer including 2D convolution, pooling, activation and batch normalizati'''
    def __init__(self,
                 bs,
                 i_ch,
                 i_h,
                 i_w,
                 k_num, # k_num == o_ch; number of kernels equal number of output channels
                 k_h,
                 k_w,
                 stride = 1, # Stride
                 zp = 0, # Zero Padding
                 mph = 2, # Pooling kernel height
                 mpw = 2, # Pooling kernel width
                 leakyrate = 0.1):
        
        self.bs = bs
        self.i_ch = i_ch
        self.i_h = i_h
        self.i_w = i_w
        self.k_num = k_num
        self.k_h = k_h
        self.k_w = k_w
        self.stride = stride
        self.zp = zp
        self.mph = mph
        self.mpw = mpw
        self.leakyrate = leakyrate
       
        # conv2d(ai, W, b) --> z
        # activation(z) --> zr
        # Batch_norm(zr, gamma, beta) --> r
        # Pool(r) --> ao, mp_trace
        self.ai = np.zeros((bs, i_ch, i_h+2*zp, i_w+2*zp), dtype='float64')
        self.z = np.zeros((bs, k_num, (i_h - k_h+1)//stride, (i_w - k_w+1)//stride), dtype='float64')
        self.bnz = np.zeros(self.z.shape, dtype='float64')
        self.r = np.zeros(self.z.shape, dtype='float64')
        self.mp_trace = np.zeros(self.r.shape, dtype='int32')
        self.ao = np.zeros((bs, k_num, (i_h - k_h+1)//stride//mph, (i_w - k_w+1)//stride//mpw), dtype='float64')
        self.W = np.random.randn(k_num, i_ch, k_h, k_w).astype('float64')
        self.b = np.random.randn(k_num, 1).astype('float64')
        self.gamma = np.ones(self.z.shape[1:], dtype='float64')
        self.beta = np.zeros(self.z.shape[1:], dtype='float64')
        
        self.dJdai = np.zeros(self.ai.shape, dtype='float64')
        self.dJdz = np.zeros(self.z.shape, dtype='float64')
        self.dJdbnz = np.zeros(self.bnz.shape, dtype='float64')
        self.dJdr = np.zeros(self.r.shape, dtype='float64')
        self.dJdao = np.zeros(self.ao.shape, dtype='float64')
        self.dJdW = np.zeros(self.W.shape, dtype='float64')
        self.dJdb = np.zeros(self.b.shape, dtype='float64')
        self.dJdgamma = np.zeros(self.gamma.shape, dtype='float64')
        self.dJdbeta = np.zeros(self.beta.shape, dtype='float64')
        
    def conv2d(self):  # (ai, W, b) --> z
        
        (i_bs, i_ch, i_rows, i_cols) = self.ai.shape # Input data
        (k_num, k_ch, k_rows, k_cols) = self.W.shape # Kernels
        (z_bs, z_ch, z_rows, z_cols) = self.z.shape   # Output z_ch == k_num
        # self.ai.fill(0.0)
        # np.copyto(self.ai[:, :, self.zp:self.zp+i_rows, self.zp:self.zp+i_cols], i_data)
        s = self.stride
        
        for b in range(z_bs): # Step over mini batch
            for c in range(z_ch): # z_ch == k_num
                for i in range(z_rows):
                    for j in range(z_cols): 
                        self.z[b, c, i, j] = np.sum(np.multiply(self.W[c, :,  :, :],
                                                                self.ai[b,
                                                                        :,
                                                                        i*s : i*s+k_rows,
                                                                        j*s : j*s+k_cols])
                                                   ) + self.b[c]
        return 
        
    
    def conv2d_prime(self): # (dJdz, W, ai) --> (dJdai, dJdW, dJdb)
        (z_bs, z_ch, z_rows, z_cols) = self.dJdz.shape
        (a_bs, a_ch, a_rows, a_cols) = self.dJdai.shape
        (k_num, k_ch, k_rows, k_cols) = self.dJdW.shape
        (b_num, b_ch) = self.dJdb.shape
        s = self.stride # self.stride
        self.dJdai.fill(0.0)
        self.dJdW.fill(0.0)
        self.dJdb.fill(0.0)
        
        # For each single dJdZ value, add a kernels to dJdai
        for b in range(z_bs):
            for c in range(z_ch):
                for i in range(z_rows):
                    for j in range(z_cols):
                        dJdz_value = self.dJdz[b, c, i, j]
                        self.dJdai[b, :, i*s:i*s+k_rows, j*s:j*s+k_cols] += dJdz_value * self.W[c, :, :, :]
        
        # Each dJdW value is a dot product of two arrays              
        for k in range(k_num):
            for c in range(k_ch):
                for i in range(k_rows):
                    for j in range(k_cols):
                        self.dJdW[k, c, i, j] = np.sum(self.ai[:, c, i:i+z_rows*s:s, j:j+z_cols*s:s] *
                                                       self.dJdz[:, k, :, :]) / self.bs 
        
        # Each dJdb value is a sum of a whole dJdZ array
        for c in range(z_ch):
            self.dJdb[c] = np.sum(self.dJdz[:, c, :, :]) / self.bs
            
        return 

    # All activation functions produce  z --> zr  
    def LeakyReLU(self): 
        np.copyto(self.r, np.where(self.z > 0, 1.0 * self.bnz, self.leakyrate * self.bnz))
        return
    
    def LeakyReLU_prime(self):
        np.copyto(self.dJdbnz, np.where(self.z > 0, 1.0 * self.dJdr, self.leakyrate * self.dJdr))
        return
    
    def tanh(self):
        np.copyto(self.r, (np.exp(self.zr) - np.exp(-self.zr)) / (np.exp(self.zr) + np.exp(-self.zr)))
        return
    
    def tanh_prime(self):
        np.copyto(self.dJdzr, (1.0 - self.r ** 2))
        return
    
    def swish(self):
        np.copyto(self.zr, self.z * self.sigmoid(self.z))
        return
    
    def swish_prime(self):
        np.copyto(self.dJdz,
                  self.zr + self.sigmoid(self.z) * (1.0 - self.zr))
        return
    
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    # Batch normalization
    # (zr, gamma, beta) --> r
    def batch_norm(self, eps=0.01):  
        
        x = self.z
        bs = x.shape[0]
        mu = 1.0/bs * np.sum(x, axis=0)[None, :, :, :]
        u = x - mu
        sigma2 = 1.0/bs * np.sum(u ** 2, axis=0)[None, :, :, :]
        q = sigma2 + eps
        v = np.sqrt(q)
        xhat = u / v
        
        np.copyto(self.bnz,
                  self.gamma * xhat + self.beta)
        
        return
    
    # (dJdr, zr, gamma) --> (dJdbeta, dJdgamma, dJdzr)
    def batch_norm_prime(self, eps = 0.01):
        
        x = self.z
        bs = x.shape[0]
        mu = 1.0/bs * np.sum(x, axis=0)[None, :, :, :]
        u = x - mu
        sigma2 = 1.0/bs * np.sum(u ** 2, axis=0)[None, :, :, :]
        q = sigma2 + eps
        v = np.sqrt(q)
        xhat = u / v
        
        self.dJdbeta = np.mean(self.dJdbnz, axis=0)[None, :, :, :]
        self.dJdgamma = np.mean(self.dJdbnz * xhat, axis=0)[None, :, :, :]
        self.dJdz = (1.0 - 1.0/bs) * (1.0/v - u**2/v**3/bs) * self.gamma * self.dJdbnz
        
        return 
    
    def batch_norm_pass(self, eps=0.01):
        
        np.copyto(self.r, self.zr)
        
        return
    
    def batch_norm_prime_pass(self, eps = 0.01):
        
        np.copyto(self.dJdzr, self.dJdr)     
        return   
    
    # Maximum Pooling
    # r --> (ao, mp_trace)
    def max_pool(self):  
        (r_bs, r_ch, r_rows, r_cols) = self.r.shape
        (a_bs, a_ch, a_rows, a_cols) = self.ao.shape
        self.mp_trace.fill(0) # record from which is the maximum so we can backprop
        
        for b in range(a_bs):
            for c in range(a_ch):
                for i in range(a_rows):
                    for j in range(a_cols):
                        pool_src2d = self.r[b,
                                            c,
                                            i*self.mph:(i+1)*self.mph,
                                            j*self.mpw:(j+1)*self.mpw] 
                        self.ao[b, c, i, j] = np.max(pool_src2d)
                        max_pos = np.unravel_index(np.argmax(pool_src2d), 
                                                   np.shape(pool_src2d))
                        self.mp_trace[b, c, i*self.mph+max_pos[0], j*self.mpw+max_pos[1]] = 1
        return
    
    def max_pool_prime(self):
        (r_bs, r_ch, r_rows, r_cols) = self.dJdr.shape
        
        for b in range(r_bs):
            for c in range(r_ch):
                for i in range(r_rows):
                    for j in range(r_cols):
                        self.dJdr[b, c, i, j] = (self.dJdao[b, c, i//self.mph, j//self.mpw] 
                                                  if self.mp_trace[b, c, i, j] == 1
                                                  else 0.0)                       
        return
    
    def forward(self):
        
        self.conv2d()  # ai --> z
        self.batch_norm()  # z --> bnz
        self.LeakyReLU()  # bnz --> r
        self.max_pool()  # r --> ao
        return
    
    def backprop(self):
        self.max_pool_prime() # dJdao --> dJdr
        self.LeakyReLU_prime() # dJdr --> dJdbnz
        self.batch_norm_prime() # dJdnz --> dJdz
        self.conv2d_prime() # dJdz --> dJdai
        
        return
    
    def update(self, lr, eta):
        # Update W, b, gamma, beta with Weight Decay
        self.W = (1.0 - eta * lr) * self.W - lr * self.dJdW
        self.b = (1.0 - eta * lr) * self.b - lr * self.dJdb
        self.gamma = (1.0 - eta * lr) * self.gamma - lr * self.dJdgamma
        self.beta = (1.0 - eta * lr) * self.beta - lr * self.beta
        return
# A Convolutional Neural Network (CNN) Consists of a number of conv layers followed by
# a number of fully-connected layers
class CNN(object):
    def __init__(self,
                input_data_spec = [10, 3, 28, 28],
                conv_layer_spec = [{"k_num" : 2,
                                    "k_h" : 3,
                                    "k_w" : 3,
                                    "stride" : 1,
                                    "zp" : 0,
                                    "mph" : 2,
                                    "mpw" :2}],
                 fc_layer_spec = [100, 50, 10]):
        
        # A list of conv layers
        self.c_net = []
        for i in range(len(conv_layer_spec)):
            self.c_net.append(Conv_Layer(bs = input_data_spec[0],
                                  i_ch = input_data_spec[1],
                                  i_h = input_data_spec[2],
                                  i_w = input_data_spec[3],
                                  k_num = conv_layer_spec[i]['k_num'],
                                  k_h = conv_layer_spec[i]['k_h'],
                                  k_w = conv_layer_spec[i]['k_w'],
                                  stride = conv_layer_spec[i]['stride'],
                                  zp = conv_layer_spec[i]['zp'],
                                  mph = conv_layer_spec[i]['mph'],
                                  mpw = conv_layer_spec[i]['mpw']))
            input_data_spec = list(self.c_net[i].ao.shape)
                       
        (bs, ch, r, c) = self.c_net[-1].ao.shape # Shape of Last Conv Layer's Output Feature Map
        fc_layer_spec[0] = ch*r*c
        self.fc_net = MLP(fc_layer_spec, BatchSize=input_data_spec[0])
        return
    
    def forward(self, input_data):
        
        # Forward through Conv Layers
        for i in range(len(self.c_net)):
            if i == 0:
                np.copyto(self.c_net[i].ai, input_data)
            else:
                np.copyto(self.c_net[i].ai, self.c_net[i-1].ao)
            self.c_net[i].forward()
            
        # Flatten the last output feature map of the conv layer for feeding to the MLP    
        input_to_fc = np.copy(self.c_net[-1].ao.reshape(self.c_net[-1].ao.shape[0], -1).T)
        
        # Forward through the MLP ( anumber of fully connected layers
        self.fc_net.forward(input_to_fc)
        
        return
    
    def backprop(self):
        
        # Backprop MLP
        self.fc_net.backprop()
        
        # Transfer the gradients from the MLB to the conv layers
        np.copyto(self.c_net[-1].dJdao, 
                  self.fc_net.net[0]['dJdi'].T.reshape(self.c_net[-1].dJdao.shape))
        
        # Backprop Conv layers
        for i in range(len(self.c_net)-1, -1, -1):
            self.c_net[i].backprop()
            if i > 0:
                np.copyto(self.c_net[i-1].dJdao, self.c_net[i].dJdai)
        return
    
    def update(self, lr, eta):
        
        # Weight update of conv layers
        for i in range(len(self.c_net)):
            self.c_net[i].update(lr, eta)
            
        # Weight Update of MLP
        self.fc_net.update(lr, eta)
        
        return

    # Find learning rate
    def proper_lr(self):
        max_W = 0.0
        max_dJdW = 0.0
        for conv_net in self.c_net:
            max_W = max(max_W, np.max(np.abs(conv_net.W)))
            max_dJdW = max(max_dJdW, np.max(np.abs(conv_net.dJdW)))
        for fc_net in self.fc_net.net[1:]:
            max_W = max(max_W, np.max(np.abs(fc_net['W'])))
            max_dJdW = max(max_dJdW, np.max(np.abs(fc_net['dJdW'])))
        return min(1.0, 0.1 * max_W / max_dJdW)
    

        
    
    # Train a CNN end-to-end
    def train(self, train_x, train_y, val_x, val_y, epoch_count, lr, eta):
        
        for e in range(epoch_count):
            # Randomly shuffle the training data at the begining of an epoch
            shuffle = np.arange(train_x.shape[0])
            np.random.shuffle(shuffle)
            train_x_s = train_x[shuffle]
            train_y_s = train_y[shuffle]
            bs = self.fc_net.bs
            
            for i in range(train_x_s.shape[0]//bs):
                x = train_x_s[i*bs:(i+1)*bs, :, : ,:]
                y = train_y_s[i*bs:(i+1)*bs]    
                self.forward(x)
                self.fc_net.loss(y, eta)
                self.backprop()
                self.update(lr, eta)
                
                print ("\nEpoch", e, "Batch", i, "J=", self.fc_net.J[-1], 
                       "Error Rate=", np.count_nonzero(np.array(y-self.fc_net.yhat)) / float(len(y)))
                      # "\nyhat=", self.fc_net.yhat, "\n   y=", y)
                print ("\nProper lr=", lr)
                print ("\nMax abs(a) of Last MLP Layer=", np.max(np.abs(self.fc_net.net[-1]['a'])), 
                       "\nMax abs(dJda) of Last MLP Layer=", np.max(np.abs(self.fc_net.net[-1]['dJda'])),
                      "\nMax abs(W) of Last MLP Layer=", np.max(np.abs(self.fc_net.net[1]['W'])), "\n")
            # Validation
            for i in range(val_x.shape[0]//bs):
                x = val_x[i*bs:(i+1)*bs, :, : ,:]
                y = val_y[i*bs:(i+1)*bs]    
                self.forward(x)
                self.fc_net.loss_val(y, eta)
                
        return             




#Build an MLP (no convolutional layer) and train it with 10 digits
f = open("mnist_test_10.csv", 'r')
a = f.readlines()
f.close()

# f = figure(figsize=(15,15));
%matplotlib inline
import matplotlib.pylab as plt
x = []
y = []
count=1
for line in a:
    linebits = line.split(',')
    x_line = [int(linebits[i]) for i in range(len(linebits))]
    x.append(x_line[1:])
    y.append(x_line[0])
    imarray = np.asfarray(linebits[1:]).reshape((28,28))
    plt.subplot(5,5,count)
    plt.subplots_adjust(hspace=0.5)
    count += 1
    plt.title("Label is " + linebits[0])
    plt.imshow(imarray, cmap='Greys', interpolation='None')
    pass


test_10_x = np.clip(np.array(x).T, 0, 1)
test_10_y = np.array(y)

Layers = (784, 50, 10)
BatchSize = 10
EpochCount = 1000
LearningRate = 0.1
Eta = 0.001

mlp10 = MLP(Layers, BatchSize)
mlp10.train(test_10_x, test_10_y, EpochCount, LearningRate, Eta)

plt.plot(mlp10.J)

mlp10.yhat

hist, bins = np.histogram(mlp10.net[-1]['dJdz'], bins=20)

width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()
np.mean(mlp10.net[-1]['a'], axis = -1)
np.var(mlp10.net[-1]['a'], axis = -1)
x = np.array([[1, 3, 5, 6, 10, 1], [3, -3, 4, 2, -2, 3], [5, -1, 4, 3, 9, 1]])
beta = np.array([[0], [0], [0]])
gamma = np.array([[1], [1], [1]])
eps = 0.01
bnx = mlp10.batch_norm(x, gamma, beta, eps)
np.mean(bnx, axis=1)



#Build an MLP and train it with 100 digits
f = open("mnist_train_100.csv", 'r')
a = f.readlines()
f.close()

# f = figure(figsize=(15,15));
%matplotlib inline
import matplotlib.pylab as plt
x = []
y = []
count=1
for line in a:
    linebits = line.split(',')
    x_line = [int(linebits[i]) for i in range(len(linebits))]
    x.append(x_line[1:])
    y.append(x_line[0])
    imarray = np.asfarray(linebits[1:]).reshape((28,28))
    plt.subplot(10,10,count)
    plt.subplots_adjust(hspace=0.5)
    count += 1
    plt.title("Label is " + linebits[0])
    plt.imshow(imarray, cmap='Greys', interpolation='None')
    pass


train_100_x = np.array(x).T
train_100_y = np.array(y)

Layers = (784, 50, 10)
BatchSize = 100
EpochCount = 1000
LearningRate = 0.1
Eta = 0.00001

mlp100 = MLP(Layers, BatchSize)
mlp100.train(train_100_x, train_100_y, EpochCount, LearningRate, Eta)

plt.plot(mlp100.J)
plt.plot(mlp100.L2_regularization)
hist, bins = np.histogram(mlp100.net[1]['dJdW'], bins=20)

width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()
np.mean(mlp100.net[-1]['z'], axis=-1)



#From here down, we train a CNN which consists of two conv layers followed by two fully connected layers
# Read in ten digits

f = open("mnist_test_10.csv", 'r')
a = f.readlines()
f.close()

# f = figure(figsize=(15,15));
%matplotlib inline
import matplotlib.pylab as plt
x = []
y = []
count=1
for line in a:
    linebits = line.split(',')
    x_line = [int(linebits[i]) for i in range(len(linebits))]
    x.append(x_line[1:])
    y.append(x_line[0])
    imarray = np.asfarray(linebits[1:]).reshape((28,28))
    plt.subplot(5,5,count)
    plt.subplots_adjust(hspace=0.5)
    count += 1
    plt.title("Label is " + linebits[0])
    plt.imshow(imarray, cmap='Greys', interpolation='None')
    pass


test_10_x = np.clip(np.array(x), 0.0, 1.0).reshape(10, 1, 28, 28)
test_10_y = np.array(y)
# Build a CNN identical to LeNet

BatchSize = 10
EpochCount = 100
LearningRate = 0.05
Eta = 0.0001

input_data_spec = [BatchSize, 1, 28, 28]
conv_layer_spec = [{"k_num" : 4, "k_h" : 3, "k_w" : 3, "stride" : 1, "zp" : 0, "mph" : 2,"mpw" :2},
                   {"k_num" : 16, "k_h" : 3, "k_w" : 3, "stride" : 1, "zp" : 0, "mph" : 2,"mpw" :2}]
fc_layer_spec = [150, 50, 10]

cnn10 = CNN(input_data_spec=input_data_spec, 
          conv_layer_spec=conv_layer_spec, 
          fc_layer_spec=fc_layer_spec)


cnn10.train(test_10_x, test_10_y, test_10_x, test_10_y, EpochCount, LearningRate, Eta)
plt.plot(cnn10.fc_net.J)
plt.plot(cnn10.fc_net.L2_regularization)
hist, bins = np.histogram(cnn10.fc_net.net[-2]['dJdb'], bins=20)

width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()
hist, bins = np.histogram(cnn10.c_net[0].dJdb, bins=20)

width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()

hist, bins = np.histogram(cnn10.c_net[0].W, bins=20)

width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()
hist, bins = np.histogram(cnn10.c_net[0].ao, bins=20)

width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()
hist, bins = np.histogram(cnn10.c_net[0].z, bins=20)

width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()

hist, bins = np.histogram(cnn10.fc_net.net[2]['dJdbeta'], bins=20)

width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()
hist, bins = np.histogram(cnn10.fc_net.net[2]['dJdW'], bins=20)

width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()
cnn100.fc_net.bs

#Now we train cnn100 with 100 digits
# Read in 100 digits

f = open("mnist_train_100.csv", 'r')
a = f.readlines()
f.close()

# f = figure(figsize=(15,15));
%matplotlib inline
import matplotlib.pylab as plt
x = []
y = []
count=1
for line in a:
    linebits = line.split(',')
    x_line = [int(linebits[i]) for i in range(len(linebits))]
    x.append(x_line[1:])
    y.append(x_line[0])
    imarray = np.asfarray(linebits[1:]).reshape((28,28))
    plt.subplot(10,10,count)
    plt.subplots_adjust(hspace=0.5)
    count += 1
    plt.title("Label is " + linebits[0])
    plt.imshow(imarray, cmap='Greys', interpolation='None')
    pass


x100 = np.array(x).reshape(100, 1, 28, 28) / 256.0 - 0.5
y100 = np.array(y)

test_100_x = x100[:, :, :, :]
test_100_y = y100[:]
val_100_x = x100[:, :, :, :]
val_100_y = y100[:]

# Build a CNN identical to LeNet

BatchSize = 25
EpochCount = 100
LearningRate = 0.1
Eta = 0.0001
LeakyRate = 0.01

input_data_spec = [BatchSize, 1, 28, 28]
conv_layer_spec = [{"k_num" : 4, "k_h" : 3, "k_w" : 3, "stride" : 1, "zp" : 0, "mph" : 2,"mpw" :2},
                   {"k_num" : 16, "k_h" : 3, "k_w" : 3, "stride" : 1, "zp" : 0, "mph" : 2,"mpw" :2}]
fc_layer_spec = [400, 50, 10]

cnn100 = CNN(input_data_spec=input_data_spec, 
          conv_layer_spec=conv_layer_spec, 
          fc_layer_spec=fc_layer_spec)

cnn100.train(test_100_x, test_100_y, val_100_x, val_100_y, EpochCount, LearningRate, Eta)
