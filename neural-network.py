import random
import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sys

#
# Shorthand:
#   "pd_" as a variable prefix means "partial derivative"
#   "d_" as a variable prefix means "derivative"
#   "_wrt_" is shorthand for "with respect to"
#   "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and
#           input to hidden layer neurons respectively
#


class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights=None, hidden_layer_bias=None,
                 output_layer_weights=None, output_layer_bias=None):

        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden(hidden_layer_weights)
        self.init_weights_from_hidden_to_output(output_layer_weights)

    # Initialize the weights from the inputs to the hidden layer
    def init_weights_from_inputs_to_hidden(self, hidden_layer_weights):
        weight_num = 0
        for neuron in self.hidden_layer.neurons:    # For every neuron in the hidden layer
            for i in range(self.num_inputs):        # Add a weight for every input
                if not hidden_layer_weights:        # If there are no weights give, pick a random weight
                    neuron.weights.append(random.random())
                else:
                    neuron.weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    # Initialize the weights from the hidden layer to the output layer
    def init_weights_from_hidden_to_output(self, output_layer_weights):
        weight_num = 0
        for output in self.output_layer.neurons:        # For every output neuron
            for neuron in self.hidden_layer.neurons:    # Add a weight for every hidden neuron
                if not output_layer_weights:
                    output.weights.append(random.random())
                else:
                    output.weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    # Run the neural net on a specific input
    def feed_forward(self, inputs):
        if len(inputs) != self.num_inputs:  # The amount of inputs should be correct
            raise ValueError
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    # Training uses online learning, i.e. updating the weights after each training instance
    def train(self, training_inputs, training_outputs):
        training_inputs = convert_to_list(training_inputs)
        training_outputs = convert_to_list(training_outputs)

        self.feed_forward(training_inputs)

        # 1. Output neuron deltas
        output_neurons = self.output_layer.neurons  # make it a little bit shorter
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(output_neurons)
        for o in range(len(self.output_layer.neurons)):
            pd_errors_wrt_output_neuron_total_net_input[o] = output_neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas
        hidden_neurons = self.hidden_layer.neurons  # make it a little bit shorter
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(hidden_neurons)
        for h in range(len(hidden_neurons)):
            # We need to calculate the pd of the error with respect to the output of each hidden layer neuron
            pd_error_wrt_output_hidden_neuron = 0
            for o in range(len(output_neurons)):
                pd_error_wrt_output_hidden_neuron += pd_errors_wrt_output_neuron_total_net_input[o] * output_neurons[o].weights[h]

            pd_errors_wrt_hidden_neuron_total_net_input[h] = pd_error_wrt_output_hidden_neuron * hidden_neurons[h].calculate_pd_output_wrt_net_input()

        # 3. Update output neuron weights
        for o in range(len(output_neurons)):
            # For every weight of this output neuron
            neuron = output_neurons[o]
            for w in range(len(output_neurons[o].weights)):
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * neuron.calculate_pd_total_net_input_wrt_weight(w)
                neuron.weights[w] -= pd_error_wrt_weight * self.LEARNING_RATE

        # 4. Update hidden neuron weights
        for h in range(len(hidden_neurons)):
            # For every weight of this hidden neuron
            neuron = hidden_neurons[h]
            for w in range(len(hidden_neurons[h].weights)):
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * neuron.calculate_pd_total_net_input_wrt_weight(w)
                neuron.weights[w] -= pd_error_wrt_weight * self.LEARNING_RATE

    # Calculates the total error on a given set of training instances
    def total_error(self, training_inputs, training_outputs):
        if len(training_inputs) != len(training_outputs):  # For every input there should be an output
            raise ValueError
        total_error = 0
        for i in range(len(training_outputs)):                  # For every instance in the training data
            training_outputs_instance = convert_to_list(training_outputs[i])
            training_inputs_instance = convert_to_list(training_inputs[i])
            self.feed_forward(training_inputs_instance)
            for o in range(len(training_outputs_instance)):           # For every output in an instance
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs_instance[o])  # Calculate the error and add it
        return total_error


class NeuronLayer:
    def __init__(self, num_neurons, bias=None):

        # The bias is the same for all neurons in a layer
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print("Neurons:", len(self.neurons))
        print("Bias: ", self.bias)
        for i in range(len(self.neurons)):
            print("  Neuron ", i)
            for w in range(len(self.neurons[i].weights)):
                print("    Weight: ", self.neurons[i].weights[w])

    # Run a feed forward on this layer with the given inputs
    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    # Get the outputs of this layer. To get accurate results, feed_forward should be called first
    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class Neuron:

    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        self.inputs = []
        self.output = 0

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(self.calculate_total_net_input(inputs))
        return self.output

    # Calculate the total net input with respect to the weights and the bias
    def calculate_total_net_input(self, inputs):
        total_net_input = 0
        for i in range(len(inputs)):
            total_net_input += inputs[i] * self.weights[i]
        return total_net_input + self.bias

    # The sigmoid activation function
    def sigmoid(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # Calculate the error for this neuron. To get an accurate result for self.output, self.feed_forward should
    # be called before this function.
    # The error function used is the mean squared error: 1/2 * (expected - real_output)^2
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) * (target_output - self.output)

    # The pd of the error function with respect to the output
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The pd of the total net input with respect to the input
    def calculate_pd_output_wrt_net_input(self):
        return self.output * (1 - self.output)

    # The pd of the total net input with respect to the input weight
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

    # How much do the total net input of the neuron need to change to move closer to the target_output
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_output_wrt_net_input()


def convert_to_list(y):
    try:
        y = [x for x in y]
    except TypeError:
        y = [y]
    return y


# Import data and scale it
#
# scale values from to values between -1 and 1
min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
df = pd.read_csv("data.csv", header=0)
# clean up data
df.columns = ["grade1", "grade2", "label"]
# remove the trailing ;
x = df["label"].map(lambda x: float(x.rstrip(';')))

# formats the input data into two arrays, one of independent variables
# and one of the dependant variable
X = df[["grade1", "grade2"]]
X = np.array(X)
X = min_max_scaler.fit_transform(X)
Y = df["label"].map(lambda x: float(x.rstrip(';')))
Y = np.array(Y)

# creating testing and training set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

net = NeuralNetwork(2, 1, 1)
training_cycles = 100000
previous_error = sys.maxsize
for i in range(training_cycles):
    for j in range(len(X_train)):
        net.train(X_train[j], Y_train[j])
    total_error = net.total_error(X_test, Y_test)
    print("Trained {} times, the total error is now: ".format(i), total_error)
    if np.abs(previous_error - total_error) < 0.000000001:
        print("converged")
        break
    previous_error = total_error
