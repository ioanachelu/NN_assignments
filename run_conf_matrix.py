import numpy as np
from network import ShallowNet
from keras.datasets import cifar10
from keras.datasets import mnist
from utils import conf_matrix

nonliniarity = "sigmoid"
dataset = "mnist"
nb_hidden_units = 500

if dataset == "cifar":
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()
  input_size = 32
  input_channels = 3
else:
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  input_size = 28
  input_channels = 1

input_dim, hidden_dim, num_clsses = input_channels * input_size * input_size, nb_hidden_units, 10
epochs = 50
batch_size = 32

model = ShallowNet(batch_size=batch_size,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_classes=num_clsses,
                    activation_fn=nonliniarity)
model.l2_reg = 0.0
learning_rate = 1e-3
lr_decay = 1
summary_interval = 100
best_params = np.load('_'.join(["model", nonliniarity, str(dataset), str(nb_hidden_units)]) + ".npy")
model.set_params(best_params)

y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

dataset_size = X_train.shape[0]

iter_per_epoch = max(int(dataset_size / batch_size), 1)
num_iterations = epochs * iter_per_epoch
epoch = 0
best_val_acc = 0
best_params = model.get_params()

val_acc = model.accuracy(X_test, y_test)
model.plot_conf_matrix(X_test, y_test)
print(val_acc)


