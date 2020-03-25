from random import random
from random import seed

from math import exp
import numpy as np

def InitializeNetwork(n_inputs, n_hidden):
	network = list()
	seed(1)
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(2)]
	network.append(output_layer)

	return network

def ActivateNeuron(weights, inputs):
	bias = weights[-1] 
	activation = bias
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]

	return activation

def Sigmoid(val):
	return 1.0 / (1.0 + exp(-val))


def ForwardPropagate(network, input):
	layer_input = input
	for layer in network:
		layer_output = []
		for neuron in layer:
			neuron['output'] = Sigmoid(ActivateNeuron(neuron['weights'], layer_input))
			layer_output.append(neuron['output'])
		layer_input = layer_output #output of previous layer will be the input of next layer
	return layer_input


def TakeDerivative(val):
	return val * (1.0 - val)

def BackwardPropagate(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer	)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * TakeDerivative(neuron['output'])

			

def UpdateWeights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

def Train(network, data, l_rate, n_epoch):
	
	for epoch in range(n_epoch):
		sum_error = 0	
		for sample in data:
			output = ForwardPropagate(network, sample)
			expected = [0 for i in range(2)]
			expected[sample[-1]] = 1
			sum_error += sum([(expected[i]-output[i])**2 for i in range(len(expected))])
			BackwardPropagate(network, expected)
			UpdateWeights(network, sample, l_rate)
		print('>Epoch=%d, Error=%.3f' % (epoch + 1, sum_error / len(data) ))


def Predict(network, data):
	predictions = list()
	for sample in data:
		output = ForwardPropagate(network, sample)
		predictions.append(output.index(max(output)))
	
	return predictions


def predict(network, row):
	outputs = ForwardPropagate(network, row)
	return outputs.index(max(outputs))