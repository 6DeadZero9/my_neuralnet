import numpy as np 
import random
from math import log10 

class Neural_network():
    """
        Neural network class
    """
    def __init__(self, input_layer, hidden_layers, output_layer):
        """
            Constructor function
            Args:
                -input_layer: integer number which defines number of the input neurons
                -hidden_layers: list which contains arbitrary number of integers each of which points on the number of neurons in a hidden layer
                -output_layer: integer number which defines number of the output neurons
            Returns: None
        """
        self.network_structure = [input_layer] + hidden_layers + [output_layer]
        self.weights = []
        self.biases = []
        self.activations = []
        self.initialization()

    def initialization(self, hidden_layer_activation='sigmoid', output_layer_activation='softmax'):
        """
            Function that initialize weights and biases for all of the connections in the network
            Args:
                -self: class instance 
                -hidden_layer_activation: name of the activation function for all of the hidden layers
                -output_layer_activation: name of the activation function for the output layer
            Returns: None
        """
        for connections in range(len(self.network_structure) - 1):
            weight_matrix = np.random.rand(self.network_structure[connections], self.network_structure[connections + 1])
            bias_matrix = np.random.rand(self.network_structure[connections + 1])
            self.weights.append(weight_matrix)
            self.biases.append(bias_matrix)
            if connections != len(self.network_structure) - 2:
                self.activations.append(self.activation_choise(connections, hidden_layer_activation))
            else:
                self.activations.append(self.activation_choise(connections, output_layer_activation)) 

    def activation_choise(self, index, activation_name):
        """
            Function that returns an activation function according to its name and the index(weather this is the output layer or not)
            Args
        """
        if index != len(self.network_structure) - 2:
            if activation_name == 'relu':
                return (activation_name, self.relu)
            elif activation_name == 'sigmoid':
                return (activation_name, self.sigmoid)
        else:
            return (activation_name, self.softmax)

    def forward_propagation(self, user_input):
        """
            Function that forward propagates through the network using users input
            Args:
                -user_input: input value provided by the user
            Returns:
                None
        """
        activated_outputs = []
        propagation_value = user_input.copy()
        for matrix_mult in range(len(self.weights)):
            dot_product = np.dot(propagation_value, self.weights[matrix_mult])
            added_biases = dot_product + self.biases[matrix_mult]
            activated = self.activations[matrix_mult][1](added_biases)
            propagation_value = activated
        return activated, activated_outputs

    def back_propagation(self, activated_outputs, predicted, expected, layer):
        if layer == len(self.weights) - 2:
            cross_entropy_loss = np.vectorize(self.cross_entropy_loss)(predicted, expected)
            print(cross_entropy_loss)
        

    def sigmoid(self, matrix_for_activation):
        """
            Sigmoid activation function that activates whole
            Args:
                -matrix_for_activation: output matrix that will be activated
            Returns:
                None
        """
        return 1 / (1 + np.exp(-matrix_for_activation))

    def relu(self, matrix_for_activation):
        """
            Relu activation function that activates whole
            Args:
                -matrix_for_activation: output matrix that will be activated
            Returns:
                None
        """
        return np.max(matrix_for_activation)

    def softmax(self, matrix_for_activation):
        """
            Softmax activation function that activates whole
            Args:
                -matrix_for_activation: output matrix that will be activated
            Returns:
                None
        """
        return np.exp(matrix_for_activation) / np.sum(np.exp(matrix_for_activation))

    def cross_entropy_loss(self, predicted, expected):
        """
            Cross entropy loss function used to calculate the loss of the network for further back propagation
            Args:
                -expected: ground truth value expected from the network to output on the given input
                -predicted: actual predicted value given by the network 
        """
        return -(expected / predicted - (1 - expected) / (1 - predicted))

    def activation_function_derivative(self, function_name):
        if function_name == 'softmax':
            pass
            

neural_net = Neural_network(3, [3, 3], 3)
forward_propagation_value, activated_outputs = neural_net.forward_propagation(np.array([1, 0, 0], np.float32))
neural_net.back_propagation(activated_outputs, forward_propagation_value, np.array([1, 0, 0], np.float32), len(neural_net.weights) - 2)
