import math

#Activation function
def sigmoid(value: float) -> float:
    value = max(-500, min(500, value))
    return 1 / (1 + math.exp(-value))

#class for a single layer in the network
class Layer:
    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        if weights is not None:
            self.weights = weights
        else:
            import random
            self.weights = [
                [random.uniform(-1, 1) for _ in n_inputs] for _ in n_neurons
            ]

        if biases is not None:
            self.biases = biases
        else: 
            biases = [0.0] * n_neurons

    def forward_prop(self, inputs):
        self.last_outputs = []
        self.inputs = inputs
        for neuron in range(len(self.weights)):
            dot_weights = sum(
                w * x for w, x in zip(self.weights[neuron], inputs)
            )
            dot_weights += self.biases[neuron]
            self.last_outputs.append(sigmoid(dot_weights))
            
            return self.last_outputs



class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        current_layer = input
        for layer in self.layers:
            current_layer = layer.forward_prop(current_layer)
        return current_layer