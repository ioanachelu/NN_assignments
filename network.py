import numpy as np
from utils import conf_matrix

class ShallowNet(object):
  def __init__(self, batch_size=10, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10, l2_reg=0.0,
               activation_fn="sigmoid"):
    self.l2_reg = l2_reg
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.num_classes = num_classes
    self.batch_size = batch_size

    fc1_w = np.random.normal(scale=(np.sqrt(2 / input_dim)), size=(input_dim, hidden_dim))
    fc1_b = np.zeros(hidden_dim)
    fc2_w = np.random.normal(scale=(np.sqrt(2 / hidden_dim)), size=(hidden_dim, num_classes))
    fc2_b = np.zeros(self.num_classes)
    self.activation_fn = activation_fn

    self.fc1 = fully_connected(w=fc1_w, b=fc1_b, activation=self.activation_fn)
    self.fc2 = fully_connected(w=fc2_w, b=fc2_b, activation=None)

    self.params = [fc1_w, fc1_b, fc2_w, fc2_b]

  def set_params(self, params):
    fc1_w, fc1_b, fc2_w, fc2_b = params
    self.fc1.w = fc1_w
    self.fc2.w = fc2_w
    self.fc1.b = fc1_b
    self.fc2.b = fc2_b

  def get_params(self):
    return [self.fc1.w, self.fc1.b, self.fc2.w, self.fc2.b]

  def logits(self, images):
    fc1_w, fc1_b, fc2_w, fc2_b = self.fc1.w, self.fc1.b, self.fc2.w, self.fc2.b

    images = images.reshape(images.shape[0], self.input_dim)
    h = self.fc1.forward(images)
    logit = self.fc2.forward(h)
    return logit

  def loss(self, images, labels):
    images = images.reshape(images.shape[0], self.input_dim)
    h = self.fc1.forward(images)
    logit = self.fc2.forward(h)

    data_loss, dlogit = softmax_cross_entropy_loss(logit, labels)
    reg_loss = self.l2_reg * np.sum(self.fc1.w ** 2)
    reg_loss += self.l2_reg * np.sum(self.fc2.w ** 2)
    loss = data_loss + reg_loss

    dx1, dfc2_w, dfc2_b = self.fc2.backward(dlogit)
    dfc2_w += self.l2_reg * self.fc2.w

    dx, dfc1_w, dfc1_b = self.fc1.backward(dx1)
    dfc1_w += self.l2_reg * self.fc1.w

    grads = [dfc1_w, dfc1_b, dfc2_w, dfc2_b]

    return loss, grads

  def accuracy(self, images, labels):
    images = images.reshape(images.shape[0], self.input_dim)
    h = self.fc1.forward(images)
    logit = self.fc2.forward(h)
    y_pred = np.argmax(logit, axis=1)
    acc = np.mean(y_pred == labels)

    return acc

  def plot_conf_matrix(self, images, labels):
    images = images.reshape(images.shape[0], self.input_dim)
    h = self.fc1.forward(images)
    logit = self.fc2.forward(h)
    y_pred = np.argmax(logit, axis=1)
    conf_matrix(y_pred, labels, self.num_classes)

    return

class fully_connected:
  def __init__(self, w, b, activation="relu"):
    self.w = w
    self.b = b
    self.act = activation

  def linear_forward(self):
    h = np.dot(self.x, self.w) + self.b
    return h

  def linear_backward(self, dout):
    dx = np.dot(dout, self.w.T)
    dw = np.dot(self.x.T, dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db

  def relu(self):
    out = np.maximum(0, self.h)
    return out

  def relu_backward(self, dout):
    dx = np.array(dout, copy=True)
    dx[self.h <= 0] = 0

    return dx

  def sigmoid(self):
    out = np.ones_like(self.h) / (1 + np.exp(-self.h))
    return out

  def sigmoid_backward(self, dout):
    out_sigmoid = np.ones_like(self.h) / (1 + np.exp(-self.h))
    dx = dout * out_sigmoid * (1 - out_sigmoid)
    return dx

  def tanh(self):
    out = (np.exp(self.h) - np.exp(-self.h)) / (np.exp(self.h) + np.exp(-self.h))
    return out

  def tanh_backward(self, dout):
    out_tanh = (np.exp(self.h) - np.exp(-self.h)) / (np.exp(self.h) + np.exp(-self.h))
    dx = dout * (1 - out_tanh ** 2)
    return dx

  def forward(self, x):
    self.x = x
    self.h = self.linear_forward()
    if self.act == "relu":
      self.h_act = self.relu()
      return self.h_act
    elif self.act == "sigmoid":
      self.h_act = self.sigmoid()
      return self.h_act
    elif self.act == "tanh":
      self.h_act = self.tanh()
      return self.h_act
    else:
      return self.h

  def backward(self, dout):
    if self.act == "relu":
      self.dh = self.relu_backward(dout)
      dx, dw, db = self.linear_backward(self.dh)
      return dx, dw, db
    elif self.act == "sigmoid":
      self.dh = self.sigmoid_backward(dout)
      dx, dw, db = self.linear_backward(self.dh)
      return dx, dw, db
    elif self.act == "tanh":
      self.dh = self.tanh_backward(dout)
      dx, dw, db = self.linear_backward(self.dh)
      return dx, dw, db
    else:
      dx, dw, db = self.linear_backward(dout)
      return dx, dw, db


def softmax_cross_entropy_loss(logit, y):
  # logit to probability
  prob = np.exp(logit - np.max(logit, axis=1)[..., None])
  prob_denominator = np.sum(prob, axis=1)[..., None]
  prob /= prob_denominator

  batch_size = prob.shape[0]
  # loss
  loss = -np.mean(np.log(prob[np.arange(batch_size), y]))
  # loss = -np.mean(prob[np.arange(batch_size), y] - np.log(prob_denominator))

  # gradient backprop
  dx = np.array(prob, copy=True)
  dx[np.arange(batch_size), y] -= 1

  dx /= batch_size

  return loss, dx
