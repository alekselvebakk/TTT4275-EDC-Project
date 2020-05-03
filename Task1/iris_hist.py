import numpy as np 
import matplotlib.pyplot as plt

# Importing the data
setosa_dataset     = np.loadtxt('./Iris_TTT4275/class_1', delimiter = ',')
versicolor_dataset = np.loadtxt('./Iris_TTT4275/class_2', delimiter = ',')
virginica_dataset  = np.loadtxt('./Iris_TTT4275/class_3', delimiter = ',')

combined_dataset = np.vstack((setosa_dataset,versicolor_dataset,virginica_dataset))

def draw_histogram(dataset_1, dataset_2, dataset_3, i_feat, title,legend_list):
  plt.hist([dataset_1[:,i_feat],dataset_2[:,i_feat],dataset_3[:,i_feat]], bins=50, color=["r","g","b"])
  plt.title(title)
  plt.legend(legend_list)
  #plt.show()

if __name__ == "__main__":
  plt.figure()
  plt.subplot(2,2,1)
  draw_histogram(setosa_dataset,versicolor_dataset,virginica_dataset, \
    0, "Histogram of sepal length", ["Setosa","Versicolor","Virginica"])
  plt.subplot(2,2,2)
  draw_histogram(setosa_dataset,versicolor_dataset,virginica_dataset, \
    1, "Histogram of sepal width", ["Setosa","Versicolor","Virginica"])
  plt.subplot(2,2,3)
  draw_histogram(setosa_dataset,versicolor_dataset,virginica_dataset, \
    2, "Histogram of petal length", ["Setosa","Versicolor","Virginica"])
  plt.subplot(2,2,4)
  draw_histogram(setosa_dataset,versicolor_dataset,virginica_dataset, \
    3, "Histogram of petal width", ["Setosa","Versicolor","Virginica"])
  plt.show()