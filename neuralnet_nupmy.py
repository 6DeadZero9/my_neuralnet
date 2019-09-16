import numpy as np 
import random
from math import log10 
#self.hidden_layers -> [5, 5, 5]

class Neural_network():

    def __init__(self, input_layer, hidden_layers, output_layer):
        self.network = []
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.network_structure = [input_layer] + hidden_layers + [output_layer]
        self.learning_rate = 0.01
        self.initialize()

    def initialize(self):
        softmax, sigmoid, relu = self.softmax, self.sigmoid, np.vectorize(self.relu)
        for init in range(len(self.network_structure) - 1):
            if init != len(self.network_structure) - 2:
                activation = sigmoid
            else:
                activation = softmax
            self.network.append([np.random.rand(self.network_structure[init], self.network_structure[init + 1]), np.random.rand(self.network_structure[init + 1]),  activation])

    def relu(self, values):
        return max(0, values)

    def sigmoid(self, values):
        return 1 / (1 + np.exp(-values))

    def softmax(self, values):
        return np.exp(values) / np.sum(np.exp(values))

    def activation_derivatives(self, layer, option, all_steps_values):
        if option == 'softmax':
            if self.output_layer == 2:
                return np.array([np.exp(np.sum(all_steps_values[layer][-1])) / np.sum(np.exp(all_steps_values[layer][-1])) ** 2 for amount in range(len(all_steps_values[layer][-1]))], np.float32)
            elif self.output_layer > 2:
                return np.array([np.exp(all_steps_values[layer][-1][index]) * (np.sum(np.exp(all_steps_values[layer][-1])) - np.exp(all_steps_values[layer][-1][index])) / np.sum(np.exp(all_steps_values[layer][-1])) for index in range(len(all_steps_values[layer][-1]))], np.float32)
        elif option == 'sigmoid':
            return (1 / (1 + np.exp(-all_steps_values[layer][-1]))) * (1 - (1 / (1 + np.exp(-all_steps_values[layer][-1]))))
        elif option == 'relu':
            pass

    def cross_entropy(self, expected, predicted, option):
        additional_bias = 1e-8
        if option == 'display':
            if self.output_layer == 2:
                return -(expected*np.log10(predicted) + (1 - expected) * np.log10(1 - predicted))
            elif self.output_layer > 2:
                return -expected*np.log10(predicted)
        elif option == 'back_propagation':
            if self.output_layer == 2:
                return -(expected / predicted - (1 - expected) / (1 - predicted))
            elif self.output_layer > 2:
                return -expected*predicted + additional_bias

    def back_propagation(self, layer, expected, predicted, all_steps_values):
        if layer == len(self.network) - 1:
            cross_entropy_derivative = self.cross_entropy(expected, predicted, 'back_propagation')
            activation_derivative = self.activation_derivatives(layer, 'softmax', all_steps_values)
            weights_values = all_steps_values[layer-1][-1].copy()
            weights_correction = np.array([[error * activation_der * connection_weight for error, activation_der in zip(cross_entropy_derivative, activation_derivative)] for connection_weight in weights_values], np.float32)
            bias_correction = np.array([error * activation_der * 1 for error, activation_der in zip(cross_entropy_derivative, activation_derivative)], np.float32)
            self.network[layer][1] = self.network[layer][1] - (bias_correction * self.learning_rate)
            self.network[layer][0] = self.network[layer][0] - (weights_correction * self.learning_rate)      
        else: 
            pass

        
    def forward_propagation(self, inputs):
        all_steps_values = []
        for layer in range(len(self.network)):
            inputs = np.dot(inputs, self.network[layer][0]) 
            inputs = inputs + self.network[layer][1]     
            not_activated = inputs.copy()
            inputs = self.network[layer][2](inputs)
            all_steps_values.append((not_activated, inputs))    
        return inputs, all_steps_values

    def __repr__(self):
        for layer in range(len(self.network)):
            if layer != len(self.network) - 1:
                layer_info = '{} Hidden'.format(layer)
            else: 
                layer_info = 'Output'
            print('\n' + layer_info, 'layer and its values:\n\tWeights\n', self.network[layer][0], '\n\tBias\n' + str(self.network[layer][1]))
        return ''

neural_net = Neural_network(2, [6, 7, 5], 2)
forward_propagation, all_steps_values = neural_net.forward_propagation(np.array([1, 0], np.float32))
neural_net.back_propagation(len(neural_net.network) - 1, np.array([1, 0], np.float32), forward_propagation, all_steps_values)
