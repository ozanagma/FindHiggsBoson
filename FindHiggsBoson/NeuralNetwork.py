from random import random
from math import exp
import numpy as np

def InitializeNetwork(n_inputs, n_hidden):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(2)]
	network.append(output_layer)

	return network

def ActivateNeuron(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]

	return activation

def Sigmoid(neuron_activation):
	return 1.0 / (1.0 + exp(-neuron_activation))


def ForwardPropagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			neuron['output'] = Sigmoid(ActivateNeuron(neuron['weights'], inputs))
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs


def TakeDerivative(output):
	return output * (1.0 - output)

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
		for j in range(len(layer)):
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

def Train(data, l_rate, n_epoch, n_hidden):
	network = InitializeNetwork(data.shape[1] - 1, n_hidden)
	for epoch in range(n_epoch):
		sum_error = 0	
		for row in data:
			i = 0
			outputs = ForwardPropagate(network, row)
			expected = [0 for i in range(2)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			BackwardPropagate(network, expected)
			UpdateWeights(network, row, l_rate)
			if(epoch == n_epoch - 1):
				predictions = np.zeros((data.shape[0], 1))
				pred_data = 1 if outputs[1] >= outputs[0] else 0
				predictions[i] = pred_data
				++i
		print('>epoch=%d, error=%.3f' % (epoch, sum_error / len(data) ))



	return predictions