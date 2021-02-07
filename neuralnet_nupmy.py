import numpy as np
#self.hidden_layers -> [5, 5, 5]

class Layer:
    def __init__(self, weights, bias, activation_function):
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

    def get_output(self, input):
        multiplicated = np.dot(self.weights, input)
        added_bias = multiplicated + self.bias

        return self.activation_function(added_bias)

class Neural_network():
    """
        Neural network class

        Args:
            input_layer: integer value that defines number of input nodes
            hidden_layers: array of integers that define number of hidden layers.
                           Integers in the array define number of nodes in
                           hidden layers.
            output_layer: integer value that defines number of output nodes
    """

    def __init__(self, input_layer, hidden_layers, output_layer):
        self.network = []
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.network_structure = [input_layer] + hidden_layers + [output_layer]
        self.learning_rate = 0.01
        self.initialize()

    def initialize(self):
        """
            Function that initializes network structure.

            Args:
                None

            Returns:
                None
        """

        for layer_index, layer in enumerate(self.network_structure):
            if not layer_index:
                continue

            current_weights = np.random.rand(self.network_structure[layer_index], self.network_structure[layer_index - 1])
            current_bias = np.random.rand(1, self.network_structure[layer_index]).transpose()
            activation_function = self.relu if layer_index != len(self.network_structure) - 1 else self.softmax

            self.network.append(Layer(current_weights, current_bias, activation_function))

    def relu(self, values):
        return np.maximum(values, 0)

    def sigmoid(self, values):
        return 1 / (1 + np.exp(-values))

    def softmax(self, values):
        return np.exp(values) / np.sum(np.exp(values))

    def forward_propagation(self, input):
        """
            Function that forward propagates input value through the network.

            Args:
                input: array with input values

            Returns:
                output: network answer for the input value

        """

        for layer_index, layer in enumerate(self.network):
            input = layer.get_output(input)

        output = input
        return output

    def __repr__(self):
        for layer_index, layer in enumerate(self.network):
            print("{}'s layer weights: \n\n{}\n\n\n".format(layer_index, layer.weights))
        return ''


if __name__ == '__main__':
    NN = Neural_network(3, [6, 7, 6], 3)
    print(NN)
    # output = NN.forward_propagation(np.array([[1, 1, 0]]).transpose())
    # print("Network output: \n{}".format(output))
