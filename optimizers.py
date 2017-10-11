import numpy as np

class GradientDescentOptimizer():
  def __init__(self, lr=0.1):
    self.lr = lr

  def update_params(self, vars, grads):
    new_vars = []
    for var, grad in zip(vars, grads):
      new_var = var - self.lr * grad
      new_vars.append(new_var)
    return new_vars

class MomentumOptimizer(GradientDescentOptimizer):
  def __init__(self, lr=1e-2, momentum=0.9):
    super(MomentumOptimizer, self).__init__(lr)
    self.momentum = momentum
    self.first_moment = None

  def update_params(self, vars, grads):
    new_vars = []
    if self.first_moment is None:
      self.first_moment = []
      for var, grad in zip(vars, grads):
        self.first_moment.append(np.zeros_like(var))

    for i, (var, grad) in enumerate(zip(vars, grads)):
      new_v = self.momentum * self.first_moment[i] - self.lr * grad
      new_var = var + new_v
      new_vars.append(new_var)
      self.first_moment[i] = new_v

    return new_vars

class RMSPropOptimizer(MomentumOptimizer):
  def __init__(self, lr=1e-2, decay=0.9, epsilon=1e-10):
    super(RMSPropOptimizer, self).__init__(lr)
    self.decay = decay
    self.epsilon = epsilon
    self.second_moment = None

  def update_params(self, vars, grads):
    new_vars = []
    if self.second_moment is None:
      self.second_moment = []
      for var, grad in zip(vars, grads):
        self.second_moment.append(np.zeros_like(var))

    for i, (var, grad) in enumerate(zip(vars, grads)):
      self.second_moment[i] = self.second_moment[i] * self.decay + \
                              (1 - self.decay) * (grad ** 2)
      new_var = var - self.lr * grad / np.sqrt(self.second_moment[i] + self.epsilon)
      new_vars.append(new_var)

    return new_vars

class AdamOptimizer(GradientDescentOptimizer):
  def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
    super(AdamOptimizer, self).__init__(lr)
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.second_moment = None
    self.first_moment = None
    self.t = 0

  def update_params(self, vars, grads):
    if self.second_moment is None:
      self.second_moment = []
      for var, grad in zip(vars, grads):
        self.second_moment.append(np.zeros_like(var))

    if self.first_moment is None:
      self.first_moment = []
      for var, grad in zip(vars, grads):
        self.first_moment.append(np.zeros_like(var))

    self.t += 1
    new_vars = []
    for i, (var, grad) in enumerate(zip(vars, grads)):
      self.first_moment[i] = self.beta1 * self.first_moment[i] + (1 - self.beta1) * grad
      self.second_moment[i] = self.beta2 * self.second_moment[i] + (1 - self.beta2) * (grad ** 2)
      f_m_hat = self.first_moment[i] / (1 - self.beta1 ** self.t)
      s_m_hat = self.second_moment[i] / (1 - self.beta2 ** self.t)
      new_var = var - self.lr * f_m_hat / np.sqrt(s_m_hat + self.epsilon)
      new_vars.append(new_var)

    return new_vars
