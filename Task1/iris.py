import numpy as np 
import matplotlib.pyplot as plt


#####Importing the data
setosa_dataset = np.loadtxt('Iris_TTT4275/Iris_TTT4275/class_1', delimiter = ',')
versicolor_dataset = np.loadtxt('Iris_TTT4275/Iris_TTT4275/class_2', delimiter = ',')
virginica_dataset = np.loadtxt('Iris_TTT4275/Iris_TTT4275/class_3', delimiter = ',')





map_for_class = {
    'Iris-setosa': np.array([1, 0, 0]),
    'Iris-versicolor': np.array([0, 1, 0]),
    'Iris-virginica': np.array([0, 0, 1])
}


def append_unit_endings(x_vector):
    rows = len(x_vector[:, 0])
    cols = len(x_vector[0, :])+1
    x_unit_ending = np.zeros([rows, cols])
    for i in range(rows):
        x_unit_ending[i, :] = np.append(x_vector[i, :], 1)
    return x_unit_ending


def square_error_single(predicted, real):
    #
    # 0.5*(predicted-real)_transposed*(predicted-real)
    error = predicted-real
    square_error = 0.5*error.T@error
    return square_error



def grad_MSE_single_value(x, g, t):
    grad_MSE_g = g-t
    grad_g_z = g*(1-g)
    grad_z_w = x
    grad_MSE_w = np.outer((grad_MSE_g*grad_g_z),grad_z_w)
    return grad_MSE_w

def sigmoid(z):
    denom = 1 + np.exp(-z)
    return 1/denom
    
def sigmoid_vector(z):
    returverdi = np.array([sigmoid(z[0]), sigmoid(z[1]), sigmoid(z[2])])
    return returverdi


def grad_MSE(W, x, t, grad):
    for i in range(len(x[:, 0])):
        z = W@x[i, :]
        g = sigmoid_vector(z)
        grad = grad + grad_MSE_single_value(x[i, :], g, t)
    return grad

def update_W(W, grad, alpha):
    return W - alpha*grad

def one_steepest_descent_step(W, x_setosa, x_versicolor, x_virginica, alpha):
    ##Making the gradient to use in the steepest descent update
    grad = 0
    #training with setosa
    grad = grad_MSE(W, x_setosa, map_for_class['Iris-setosa'], grad)
    #training with versicolor
    grad = grad_MSE(W, x_versicolor, map_for_class['Iris-versicolor'], grad)
    #training with virginica
    grad = grad_MSE(W, x_virginica, map_for_class['Iris-virginica'], grad)
    W = update_W(W, grad, alpha)

    return W

def predict(W, x):
    z = W@x
    g = sigmoid_vector(z)
    return g


def summed_square_error_single_type(W, x, t):
    sum = 0
    for i in range(len(x[:, 0])):
        g = predict(W, x[i, :])
        sum = sum + square_error_single(g, t)
    return sum


def summed_square_error(W, x_setosa, x_versicolor, x_virginica):
    sum = 0
    sum = sum + summed_square_error_single_type(W, x_setosa, map_for_class['Iris-setosa'])
    sum = sum + summed_square_error_single_type(W, x_versicolor, map_for_class['Iris-versicolor'])
    sum = sum + summed_square_error_single_type(W, x_virginica, map_for_class['Iris-virginica'])
    return sum

def check_match(predicted, real):
    if np.array_equal(real, map_for_class['Iris-setosa']):
        predicted_largest = np.amax(predicted)
        predicted_value = predicted[0]
        return (predicted_largest == predicted_value)
    elif np.array_equal(real, map_for_class['Iris-versicolor']):
        predicted_largest = np.amax(predicted)
        predicted_value = predicted[1]
        return (predicted_largest == predicted_value)
    else:
        predicted_largest = np.amax(predicted)
        predicted_value = predicted[2]
        return (predicted_largest == predicted_value)


def error_rate_single_class(W, x, t):
    errors = 0
    for i in range(len(x[:, 0])):
        g = predict(W, x[i, :])
        if not check_match(g, t):
            errors = errors + 1
    return errors

def error_rate(W, x_setosa, x_versicolor, x_virginica, tests):
    errors = 0
    errors = errors + error_rate_single_class(W, x_setosa, map_for_class['Iris-setosa'])
    errors = errors + error_rate_single_class(W, x_versicolor, map_for_class['Iris-versicolor'])
    errors = errors + error_rate_single_class(W, x_virginica, map_for_class['Iris-virginica'])
    return errors/(3*tests)

def confusion_matrix(W, setosa_data, versicolor_data, virginica_data):
    confuse = np.zeros([3, 3])
    for i in range(len(setosa_data)):
        g = predict(W, setosa_data[i, :])
        if check_match(g, map_for_class['Iris-setosa']):
            confuse[0, 0] = confuse[0, 0] + 1
        elif check_match(g, map_for_class['Iris-versicolor']):
            confuse[0, 1] = confuse[0, 1] + 1
        elif check_match(g, map_for_class['Iris-virginica']):
            confuse[0, 2] = confuse[0, 2] + 1
        else:
            print("\nDidnt classify as anything, value:\n", g)
    for i in range(len(versicolor_data)):
        g = predict(W, versicolor_data[i, :])
        if check_match(g, map_for_class['Iris-versicolor']):
            confuse[1, 1] = confuse[1, 1] + 1
        elif check_match(g, map_for_class['Iris-virginica']):
            confuse[1, 2] = confuse[1, 2] + 1
        elif check_match(g, map_for_class['Iris-setosa']):
            confuse[1, 0] = confuse[1, 0] + 1
        else:
            print("\nDidnt classify as anything, value:\n", g)
    for i in range(len(virginica_data)):
        g = predict(W, virginica_data[i, :]) 
        if check_match(g, map_for_class['Iris-virginica']):
            confuse[2, 2] = confuse[2, 2] + 1
        elif check_match(g, map_for_class['Iris-setosa']):
            confuse[2, 0] = confuse[2, 0] + 1
        elif check_match(g, map_for_class['Iris-versicolor']):
            confuse[2, 1] = confuse[2, 1] + 1
        else:
            print("\nDidnt classify as anything, value:\n", g)
    return confuse
#####################Part 1#####################:

#####Formating data so x = [x1, x2, x3, x4, 1]
setosa_dataset_ones = append_unit_endings(setosa_dataset)
versicolor_dataset_ones = append_unit_endings(versicolor_dataset)
virginica_dataset_ones = append_unit_endings(virginica_dataset)



training_samples = 30
tests = 20
iterations = 3000
alpha = 0.008


##   a) Using first 30 samlpes as training and last 20 as testing
setosa_training = setosa_dataset_ones[:training_samples, :]
versicolor_training = versicolor_dataset_ones[:training_samples, :]
virginica_training = virginica_dataset_ones[:training_samples, :]

setosa_testing = setosa_dataset_ones[training_samples:, :]
versicolor_testing= versicolor_dataset_ones[training_samples:, :]
virginica_testing = virginica_dataset_ones[training_samples:, :]

#Initializing arrays and variables
W = np.array([  [0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0] ])

square_error = np.zeros(iterations)
rate_of_error = np.zeros(iterations)

#Training sequence
for i in range(iterations):
    W= one_steepest_descent_step(W,setosa_training, versicolor_training, virginica_training, alpha)
    square_error[i] = summed_square_error(W, setosa_testing, versicolor_testing, virginica_testing)
    rate_of_error[i] = error_rate(W, setosa_testing, versicolor_testing, virginica_testing, tests)

confuse_training = confusion_matrix(W, setosa_training, versicolor_training, virginica_training)
confuse_test = confusion_matrix(W, setosa_testing, versicolor_testing, virginica_testing)
print("\nConfusion matrix for training: \n", confuse_training)
print("\nConfusion matrix for testing: \n", confuse_test)

plt.subplot(211)
plt.plot(rate_of_error)
plt.xlabel("Iterations")
plt.ylabel("Error rate training with 30 first")

##   d) Using last 30 samlpes as training and first 20 as testing
setosa_training = setosa_dataset_ones[tests:, :]
versicolor_training = versicolor_dataset_ones[tests:, :]
virginica_training = virginica_dataset_ones[tests:, :]

setosa_testing = setosa_dataset_ones[:tests, :]
versicolor_testing= versicolor_dataset_ones[:tests, :]
virginica_testing = virginica_dataset_ones[:tests, :]

#Initializing arrays and variables
W_2 = np.array([    [0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0] ])

square_error_2 = np.zeros(iterations)
rate_of_error_2 = np.zeros(iterations)
#Training sequence
for i in range(iterations):
    W_2= one_steepest_descent_step(W_2,setosa_training, versicolor_training, virginica_training, alpha)
    square_error_2[i] = summed_square_error(W_2, setosa_testing, versicolor_testing, virginica_testing)
    rate_of_error_2[i] = error_rate(W_2, setosa_testing, versicolor_testing, virginica_testing, tests)

confuse_training_2 = confusion_matrix(W_2, setosa_training, versicolor_training, virginica_training)
confuse_test_2 = confusion_matrix(W_2, setosa_testing, versicolor_testing, virginica_testing)
print("\nConfusion matrix for training: \n", confuse_training_2)
print("\nConfusion matrix for testing: \n", confuse_test_2)


plt.subplot(212)
plt.plot(rate_of_error_2)
plt.xlabel("Iterations")
plt.ylabel("Error rate training with 30 last")






###########Part 2:







plt.show()