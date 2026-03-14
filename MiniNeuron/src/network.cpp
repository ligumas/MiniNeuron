#include "network.h"
#include "Layer.h"
#include <vector>
#include <iostream>
#include <assert.h>

namespace MiniNeuron {

	//add layer to the back of the network. also prints a debug text.
	void network::add(Layer&& x) {
		layers.push_back(x);
		std::cout << "added layer to network" << std::endl;
	}

	//initialize layers with random values. currently supports only he init automatically but will be expanded
	void network::initlayers() {
		for (size_t i = 0; i < layers.size(); i++) {
			layers[i].initWeights();
		}
	}

	//prints all layers and shows thier neuron count. used for debug, to be changed.
	void network::printNetwork() {
		for (size_t i = 0; i < layers.size(); i++) {
			std::cout << "Layer(" << i << ")" << " neurons(" << layers[i].NeuronCount << ")" << std::endl;
		}
	}

	//executes every layer in the network and returns the last layer(input layer) output
	std::vector<float> network::forward(std::vector<float> inputs) {
		std::vector<float> x = inputs;
		for (size_t i = 0; i < layers.size(); i++) {
			x = layers[i].Forward(x);
		}
		return x;
	}

	//score the overall network perfomance, used to tell user how good the network is.
	float network::loss(const std::vector<float>& p, const std::vector<float>& y, LossTypes losstype) {
		float loss = 0.0f;
		for (size_t i = 0; i < p.size(); i++) {

			switch (losstype) {
				case LossTypes::crossEntropy:
				{
					loss += -y[i] * std::log(p[i] + 1e-8f);
					break;
				}
				case LossTypes::MSE:
				{
					float diff = y[i] - p[i];
					loss += 0.5f * (diff * diff);
					break;
				}
			}
		}
		if (losstype == LossTypes::MSE) {
			return loss / p.size();
		}
		return loss;
	}

	//score the target and wanted output. so used only once in backpropagation.
	float network::loss_derivative(float p, float y, LossTypes losstype) {
		switch (losstype) {
			case LossTypes::MSE:
			{
				return p - y;
			}
			case LossTypes::crossEntropy:
			{
				return -y / (p + 1e-8f);
			}
			default:
				return 0.0f;
		}
	}

	void network::backpropagate(const std::vector<float>& targets) {
		for (int i = (int)layers.size() - 1; i >= 0; i--) {
			bool isOutput = (i == layers.size() - 1);

			if (isOutput) {
				layers[i].backpropagation(targets, {}, true);
			} else {
				layers[i].backpropagation(layers[i + 1].Delta, layers[i + 1].weights, false);
			}
		}
	}

	void network::update_network(const std::vector<float>& og_inputs, float learningRate) {
		for (size_t i = 0; i < layers.size(); i++) {
			if (i == 0) {
				layers[i].updateWeights(og_inputs, learningRate);
			}else {
				layers[i].updateWeights(layers[i - 1].result, learningRate);
			}
		}
	}

	float network::epoch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, float learning_rate) {
		float total_loss = 0;
		assert(inputs.size() == targets.size());
		for (size_t i = 0; i < inputs.size(); i++)
		{
			const std::vector<float>& x = inputs[i];
			const std::vector<float>& y = targets[i];

			std::vector<float> prediction = forward(x);
			total_loss += loss(prediction, y, LossTypes::MSE);
			backpropagate(y);
			update_network(x, learning_rate);
		}
		return total_loss / inputs.size();
	}

	void network::train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets,int epochs, float learning_rate) {
		std::cout << "Starting Training..." << std::endl;

		

		for (int i = 0; i < epochs; i++) {
			float total_loss = 0;
			float loss = 0;

			loss = epoch(inputs, targets, learning_rate);
			total_loss += loss;

			if (i % 10 == 0) {
				std::cout.precision(8);
				std::cout << std::fixed;
				std::cout << "Epoch: " << i
					<< " | Loss: " << loss
					<< " | Avg Loss: " << total_loss / 4.0f
					<< std::endl;
			}
			std::cout.unsetf(std::ios::fixed);
		}
	}

}