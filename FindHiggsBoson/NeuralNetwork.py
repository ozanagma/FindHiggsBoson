from random import random
from random import seed

from math import exp
from math import tanh
import numpy as np

def InitializeNetwork(n_inputs, n_hidden_1, n_hidden_2):
	network = list()
	hidden_layer_1	= [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden_1)]
	network.append(hidden_layer_1)
	hidden_layer_2	= [{'weights':[random() for i in range(n_hidden_1 + 1)]} for i in range(n_hidden_2)]
	network.append(hidden_layer_2)
	#hidden_layer_3	= [{'weights':[random() for i in range(n_hidden_1 + 1)]} for i in range(n_hidden_3)]
	#network.append(hidden_layer_3)
	output_layer	= [{'weights':[random() for i in range(n_hidden_2 + 1)]} for i in range(2)]
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

def Relu(val):
   return max(0.0, val)

def LeakyRelu(val):
   return max(0.01 * val, val)

def Tanh(val):
	return (exp(val)-exp(-val))/(exp(val)+exp(-val))

def ForwardPropagate(network, input):
	layer_input = input
	for layer in network:
		layer_output = []
		for neuron in layer:
			neuron['output'] = Sigmoid(ActivateNeuron(neuron['weights'], layer_input))
			layer_output.append(neuron['output'])
		layer_input = layer_output #output of previous layer will be the input of next layer
	return layer_input


def SigmoidDerivative(val):
	return val * (1.0 - val)

def ReluDerivative(val):
	return 0.0 if val <= 0 else 1

def LeakyReluDerivative(val):
	return 0.01 if val <= 0 else 1

def TanhDerivative(val):
	t=(exp(val)-exp(-val))/(exp(val)+exp(-val))
	return 1- t**2

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
			neuron['delta'] = errors[j] * SigmoidDerivative(neuron['output'])

			

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
			sum_error += sum([(sample[-1]-output[sample[-1]])**2]) / len(data)
			BackwardPropagate(network, expected)
			UpdateWeights(network, sample, l_rate)
		print('>Epoch=%d, Error=%.3f' % (epoch + 1, sum_error))


def Predict(network, data):
	predictions = list()
	for sample in data:
		output = ForwardPropagate(network, sample)
		predictions.append(output.index(max(output)))
	
	return predictions


def predict(network, row):
	outputs = ForwardPropagate(network, row)
	return outputs.index(max(outputs))