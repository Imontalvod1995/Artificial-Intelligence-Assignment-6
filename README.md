# Artificial-Intelligence-Assignment-6

Author: Ivan Montalvo

Language: Python 3 (NumPy only)

## What this is
A from-scratch feedforward neural network (no ML libraries like scikit-learn, TensorFlow, or PyTorch) trained to classify flowers in the classic Iris dataset into one of three species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

Everything — feedforward, backpropagation, weight updates — is implemented by hand using NumPy arrays.

## How it works

1. Data loading (`transform_iris_data`)

Reads a file called `iris_data.txt` (comma-separated values: 4 numeric features + a class name `like Iris-setosa`). Converts the class names into integers (0, 1, 2) using a `CLASS_MAPPING` dictionary, and returns features (`X`) and labels (`y`) as NumPy arrays.

2. Train/Val/Test split (`split_data`)

Shuffles the dataset and splits it into training (60%), validation (20%), and test (20%) sets by default.

3. Neural network (`NeuralNetwork` class)

A simple network with:

- 4 input features
- 2 hidden layers (8 neurons each)
- 3 output neurons (one per species)
- Sigmoid activation on every layer

It includes:

- `feedforward()`: pushes data through the network
- `backpropagation()`: computes gradients and updates weights/biases using plain gradient descent
- `train()`: runs feedforward + backprop for a set number of epochs, printing loss and validation accuracy every 10 epochs
- `validate()`: computes classification accuracy on a given dataset

4. Main program (main)

- Loads and splits the data
- Normalizes features by dividing by the max value in each column
- Converts training labels to one-hot vectors
- Trains the network for 500 epochs (learning rate 0.1)
- Prints final accuracy on the test set
- Drops into an interactive loop: you type in 4 comma-separated flower measurements (sepal length, sepal width, petal length, petal width) and it prints the predicted species. Type `q` to quit.

## How to run it

- Make sure you have Python 3 and NumPy installed (`pip install numpy`).
- Put the `iris_data.txt` file (comma-separated Iris dataset, no header row, blank lines allowed) in the same folder as the script.
- Run `python Assignment_6_Ivan_Montalvo.py`
- Watch the training logs, check the test accuracy, then try entering your own flower measurements when prompted.
