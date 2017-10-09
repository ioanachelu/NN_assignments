class sgd_momentum()
def sgd_momentum(w, dw, config=None):
  if config is None:
    config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)
  v = config.get('velocity', np.zeros_like(w))
  next_v = config['momentum'] * v - config['learning_rate'] * dw
  next_w = w + next_v

  ##########################################################################
  # TODO: Implement the momentum update formula. Store the updated value in   #
  # the next_w variable. You should also use and update the velocity v.       #
  ##########################################################################
  pass
  ##########################################################################
  #                             END OF YOUR CODE                              #
  ##########################################################################
  config['velocity'] = next_v

  return next_w, config