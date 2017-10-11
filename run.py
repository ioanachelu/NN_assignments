import numpy as np
from network import ShallowNet
from keras.datasets import cifar10
from keras.datasets import mnist
from utils import conf_matrix
import optimizers
import matplotlib.pyplot as plt

nonliniarity = "tanh"
dataset = "mnist"
nb_hidden_units = 500
losses = []

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
# checkpoint_interval = 1
optimizer = optimizers.AdamOptimizer(learning_rate)

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

for t in range(num_iterations):
  indices = np.random.choice(dataset_size, batch_size)
  X_batch = X_train[indices]
  y_batch = y_train[indices]
  loss, grads = model.loss(X_batch, y_batch)

  new_params = model.get_params()
  new_params = optimizer.update_params(new_params, grads)
  # for var, grad in zip(new_params, grads):
  #   var -= learning_rate * grad
  model.set_params(new_params)

  if (t + 1) % iter_per_epoch == 0:
    epoch += 1
    learning_rate *= lr_decay

  if (t + 1) % summary_interval == 0:
    train_acc = model.accuracy(X_batch, y_batch)
    val_acc = model.accuracy(X_test, y_test)
    # model.plot_conf_matrix(X_test, y_test)
    print("Epoch {} >>>  \n Iter {} / {} >>> lr: {} >>> Loss: {} >>> train_acc: {} >>> val_acc: {}".format(epoch, t + 1, num_iterations,
                                                                                                   learning_rate, loss,
                                                                                                   train_acc, val_acc))
    losses.append(val_acc)
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      best_params = []
      for p in model.get_params():
        best_params.append(p.copy())

model.set_params(best_params)
np.save('_'.join(["model", nonliniarity, str(dataset), str(nb_hidden_units)]), model.params)
plt.plot(losses)
plt.savefig('_'.join(["model", nonliniarity, str(dataset), str(nb_hidden_units)]) + "_adam.png")


