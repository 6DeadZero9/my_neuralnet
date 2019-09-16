import numpy as np 
import random
from math import exp
#self.hidden_layer -> {'0': 4, '1': 5, '2': 4}

class Neural_network():

    def __init__(self, input_layer, hidden_layer, output_layer):
        self.network = {}
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.initialize()

    def initialize(self):
        hidden_layers = {}
        for index, layer in enumerate(sorted(self.hidden_layer.keys())):
            hidden_layers[str(index)] = {}
            for neuron in range(self.hidden_layer[layer]):
                if layer == '0':
                    hidden_layers[str(index)][str(neuron)] = {'weights': [random.random() for input_value in range(self.input_layer)], 'bias': random.random()}
                else: 
                    hidden_layers[str(index)][str(neuron)] = {'weights': [random.random() for input_value in range(self.hidden_layer[str(int(layer)-1)])], 'bias': random.random()}

        last_hidden_layer = sorted(hidden_layers.keys())[-1]
        last_hidden_layer = len(hidden_layers[last_hidden_layer].keys())
        output_layer = {}
        for neuron in range(self.output_layer):
            output_layer[str(neuron)] = {'weights': [random.random() for input_value in range(last_hidden_layer)], 'bias': random.random()}

        self.network['hidden_layers'] = hidden_layers
        self.network['output_layer'] = output_layer

    def sigmoid(self, input_value):
        return 1 / (1 + exp(-input_value))

    def softmax(self, input_array):
        return [exp(x) / sum([exp(y) for y in input_array]) for x in input_array]

    def forward_propagation(self, inputs):
        for layer in sorted(self.network['hidden_layers'].keys()):
            new_inputs = []
            for neuron in sorted(self.network['hidden_layers'][layer].keys()):
                new_inputs.append(self.sigmoid(sum([weight * inpt for weight, inpt in zip(self.network['hidden_layers'][layer][neuron]['weights'], inputs)]) + self.network['hidden_layers'][layer][neuron]['bias'])) 
            inputs = new_inputs[:]
        outputs = []
        for neuron in sorted(self.network['output_layer'].keys()):
            outputs.append(self.sigmoid(sum([weight * inpt for weight, inpt in zip(self.network['output_layer'][neuron]['weights'], inputs)]) + self.network['output_layer'][neuron]['bias'])) 
        return self.softmax(outputs)

    def __repr__(self):
        for layer in sorted(first_neuralnet.network['hidden_layers'].keys()):
            print('Hidden layer number: ', layer)
            for neuron in sorted(first_neuralnet.network['hidden_layers'][layer].keys()):
                print('\tNeuron number: ', neuron, ' and its values: ', first_neuralnet.network['hidden_layers'][layer][neuron])
            print()
        print('Output layer')
        for neuron in sorted(first_neuralnet.network['output_layer'].keys()):
                print('\tNeuron number: ', neuron, ' and its values: ', first_neuralnet.network['output_layer'][neuron])


first_neuralnet = Neural_network(2, {'0': 5, '1': 5, '2': 5}, 2)
print(first_neuralnet.forward_propagation([1, 0]))
