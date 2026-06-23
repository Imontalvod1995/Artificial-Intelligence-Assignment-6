####################
#                  #
#   Assignment 6   #
#                  #
#   Ivan Montalvo  #
#                  #
#################### 

# Library imports and global variables:

import numpy as np

np.random.seed(42)
rng = np.random.default_rng()
CLASS_MAPPING = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

# Function definition:

def transform_iris_data():

    """ This function opens the dataset, splits it into features and labels
    and shapes them into an array. """

    X = []
    y = []
    
    with open('iris_data.txt', 'r') as f:
        data = [line.strip().split(',') for line in f.readlines()]

    for row in data:

        if row != ['']:

            features = list(map(float, row[:-1])) # features in float form
            label = CLASS_MAPPING[row[-1]]  # Labels in int format
            X.append(features)
            y.append(label)
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)  # Vector column

    return X, y

def split_data(X, y, train_size = 0.6, val_size = 0.2):
    
    """ This function splits the data into train, validate, and test. """
    
    # Shuffle the data
    
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)

    # Split the data

    train_index = int(train_size * len(indices))
    val_index = int(val_size * len(indices)) + train_index
    train_indices = indices[:train_index]
    val_indices = indices[train_index:val_index]
    test_indices = indices[val_index:]

    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
    y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test

def sigmoid(x):

    """ This function defines the sigmoid function. """

    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    
    """ This function defines the sigmoid derivative function. """

    return x * (1 - x)

# Class definition: 

class NeuralNetwork:

    """ This class sets up all the necessary functions to create the Neural Network. """

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        
        """ This initialization function sets the amount of layers and their 
        connectivities of the NN. """
        
        self.weights_input_hidden1 = rng.random((input_size, hidden_size_1))
        self.bias_hidden1 = rng.random(hidden_size_1)

        self.weights_hidden1_hidden2 = rng.random((hidden_size_1, hidden_size_2))
        self.bias_hidden2 = rng.random(hidden_size_2)

        self.weights_hidden2_output = rng.random((hidden_size_2, output_size))
        self.bias_output = rng.random(output_size)

    def feedforward(self, X):

        """ This function allows to perform the feed forward operation. """

        self.hidden_layer1_input = np.dot(X, self.weights_input_hidden1) + self.bias_hidden1
        self.hidden_layer1_output = sigmoid(self.hidden_layer1_input)

        self.hidden_layer2_input = np.dot(self.hidden_layer1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        self.hidden_layer2_output = sigmoid(self.hidden_layer2_input)
        
        self.output_layer_input = np.dot(self.hidden_layer2_output, self.weights_hidden2_output) + self.bias_output
        self.output = sigmoid(self.output_layer_input)
        
        return self.output

    def backpropagation(self, X, y, learning_rate):

        """ This function performs the backpropagation, using the values from feedforward. """

        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        hidden2_error = np.dot(output_delta, self.weights_hidden2_output.T)
        hidden2_delta = hidden2_error * sigmoid_derivative(self.hidden_layer2_output)

        hidden1_error = np.dot(hidden2_delta, self.weights_hidden1_hidden2.T)
        hidden1_delta = hidden1_error * sigmoid_derivative(self.hidden_layer1_output)

        self.weights_hidden2_output += np.dot(self.hidden_layer2_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        
        self.weights_hidden1_hidden2 += np.dot(self.hidden_layer1_output.T, hidden2_delta) * learning_rate
        self.bias_hidden2 += np.sum(hidden2_delta, axis=0) * learning_rate

        self.weights_input_hidden1 += np.dot(X.T, hidden1_delta) * learning_rate
        self.bias_hidden1 += np.sum(hidden1_delta, axis=0) * learning_rate

    def train(self, X, y, X_val, y_val, epochs, learning_rate):

        """ This function performs the training using feedforward and backpropagation, 
        as well as the validate. """

        for epoch in range(epochs):
            
            self.feedforward(X)
            self.backpropagation(X, y, learning_rate)
            
            if epoch % 10 == 0:
                
                loss = np.mean((y - self.output) ** 2)
                uracy = self.validate(X_val, y_val)
                
                print(f"Epoch {epoch}, Loss: {loss}, Accuracy on validation data: {uracy * 100:.2f}%")
    
    def validate(self, X, y):

        """ This function checks the values and performs validation. """

        predictions = self.feedforward(X)
        predictions = np.clip(predictions, 0, 1)
        class_pred = np.argmax(predictions, axis = 1)
        true_class = y.flatten()
        acc = np.mean(class_pred == true_class)

        return acc
        
# Running the code:

def main():

    """ This is the main function that initializes the NN, provides the accuracy result,
    and the while loop that allows for interactivity with the user. """

    X, y = transform_iris_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    nn = NeuralNetwork(input_size=4, hidden_size_1 = 8, hidden_size_2 = 8, output_size=3)

    X_train = X_train / np.max(X_train, axis=0)
    X_val = X_val / np.max(X_val, axis = 0)
    X_test = X_test / np.max(X_test, axis=0)

    y_train_one_hot = np.zeros((y_train.size, y_train.max() + 1))
    y_train_one_hot[np.arange(y_train.size), y_train.flatten()] = 1

    nn.train(X_train, y_train_one_hot, X_val, y_val, epochs = 500, learning_rate = 0.1)

    predictions = nn.feedforward(X_test)
    predictions = np.clip(predictions, 0, 1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = y_test.flatten()

    # Calculate accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")

    # User interactivity while loop 
    while True:

        user_input = input("Enter flower features (sepal length, sepal width, petal length, petal width) or 'q' to quit: ")
        
        if user_input.lower() == 'q':
            break
        
        try:
            features = np.array(list(map(float, user_input.split(',')))).reshape(1, -1)
            features = features / np.max(features, axis=1)
            prediction = nn.feedforward(features)
            prediction = np.clip(prediction, 0, 1)
            predicted_class = np.argmax(prediction)
            
            print(f"Predicted class name: {list(CLASS_MAPPING.keys())[predicted_class]}")
        
        except Exception as e:
            print(f"Error: {e}. Please enter valid numeric values.")

if __name__ == "__main__":
