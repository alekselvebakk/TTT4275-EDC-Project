import numpy as np 
import matplotlib.pyplot as plt

import iris as task1 # script from task1 for re-use of functions

# Importing the data
setosa_raw     = np.loadtxt('./Iris_TTT4275/class_1', delimiter = ',')
versicolor_raw = np.loadtxt('./Iris_TTT4275/class_2', delimiter = ',')
virginica_raw  = np.loadtxt('./Iris_TTT4275/class_3', delimiter = ',')

def slice_from_dataset(dataset, indices):
  matrix_to_keep = 0
  first_column_flag = False
  for i in range(dataset.shape[1]):
    if i in indices: continue
    if first_column_flag:
      matrix_to_keep = np.hstack((matrix_to_keep, np.reshape(dataset[:,i],(50,1))))
    else:
      first_column_flag = True
      matrix_to_keep = np.reshape(dataset[:,i],(50,1))
  return matrix_to_keep


def train_and_test_classifier(setosa_set, versicolor_set, virginica_set, \
                              alpha, iterations, training_samples=30):
  
  # Splitting into train and test subsets
  setosa_train     = setosa_set[:training_samples, :]
  versicolor_train = versicolor_set[:training_samples, :]
  virginica_train  = virginica_set[:training_samples, :]

  setosa_test      = setosa_set[training_samples:, :]
  versicolor_test  = versicolor_set[training_samples:, :]
  virginica_test   = virginica_set[training_samples:, :]

  # Initializing arrays and variables
  n_classes  = 3
  n_features = setosa_set.shape[1]
  W = np.zeros((n_classes,n_features))

  # Training sequence
  for i in range(iterations):
      W = task1.one_steepest_descent_step( W, setosa_train, \
          versicolor_train, virginica_train, alpha)

  confuse_training = task1.confusion_matrix(W, setosa_train, versicolor_train, virginica_train)
  confuse_test = task1.confusion_matrix(W, setosa_test, versicolor_test, virginica_test)
  print("Confusion matrix, train: \n", confuse_training)
  print("Confusion matrix, test: \n", confuse_test)

if __name__ == "__main__":
  # Formatting data s.t. x = [x1, x2, x3, x4, 1]
  setosa_full     = task1.append_unit_endings(setosa_raw)
  versicolor_full = task1.append_unit_endings(versicolor_raw)
  virginica_full  = task1.append_unit_endings(virginica_raw)

  # Configuration & Tuning
  training_samples = 30
  tests = 20
  iterations = 3000
  alpha = 0.008

  print("Test of classifier without first feature:")  
  train_and_test_classifier(slice_from_dataset(setosa_full, [0]), \
                            slice_from_dataset(versicolor_full, [0]), \
                            slice_from_dataset(virginica_full, [0]),\
                            alpha, iterations, 30)

  print("\n\nTest of classifier without second feature:")
  train_and_test_classifier(slice_from_dataset(setosa_full, [1]), \
                            slice_from_dataset(versicolor_full, [1]), \
                            slice_from_dataset(virginica_full, [1]),\
                            alpha, iterations, 30)

  print("\n\nTest of classifier without both sepal features:")
  train_and_test_classifier(slice_from_dataset(setosa_full, [0,1]), \
                            slice_from_dataset(versicolor_full, [0,1]), \
                            slice_from_dataset(virginica_full, [0,1]),\
                            alpha, iterations, 30)

  print("\n\nTest of classifier with only fourth feature:")
  train_and_test_classifier(slice_from_dataset(setosa_full, [0,1,2]), \
                            slice_from_dataset(versicolor_full, [0,1,2]), \
                            slice_from_dataset(virginica_full, [0,1,2]),\
                            alpha, iterations, 30)

  

  