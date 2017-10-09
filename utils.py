import numpy as np
from matplotlib import pyplot as plt

def conf_matrix(y_pred, y, num_classes):
  # confusion matrix
  conf_mat = np.zeros((num_classes, num_classes))
  num_batches = y.shape[0]
  for i in range(num_batches):
    conf_mat[y[i], y_pred[i]] += 1

  plot_confusion_matrix(conf_mat)

def plot_confusion_matrix(df_confusion):
  plt.matshow(df_confusion, cmap=plt.cm.gray_r)  # imshow
  plt.title("Confusion matrix")
  plt.colorbar()
  plt.ylabel("Actual")
  plt.xlabel("Predicted")
  plt.show()


# y = np.array([0, 1, 5, 3, 1])
# y_pred = np.array([2, 1, 4, 3, 1])
# num_classes = 10
# num_batches = y.shape[0]
# conf_matrix = np.zeros((10, 10))
# for i in range(num_batches):
#     conf_matrix[y[i], y_pred[i]] += 1
#
# print (conf_matrix)
# plot_confusion_matrix(conf_matrix)
# print("sdasds")