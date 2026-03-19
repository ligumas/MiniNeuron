#include "network.h"
#include "Layer.h"
#include <vector>
#include <iostream>
#include <cassert>

namespace MiniNeuron {

	//add layer to the back of the network. also prints a debug text.
	void Network::add(Layer&& x) {
		layers.push_back(std::move(x));
		//std::cout << "added layer to network" << std::endl;
	}

	//initialize layers
	void Network::initLayers() {
		for (size_t i = 0; i < layers.size(); i++) {
			layers[i].initLayer();
		}
	}

	//prints all layers and shows thier neuron count. used for debug, to be changed.
	void Network::printNetwork() {
		for (size_t i = 0; i < layers.size(); i++) {
			std::cout << "Layer(" << i << ")" << " neurons(" << layers[i].getNeuronCount() << ")" << std::endl;
		}
	}

	//used to predict/use the neural network
	std::vector<float> Network::forward(const std::vector<float>& inputs) {
		std::vector<float> x = inputs;
		for (size_t i = 0; i < layers.size(); i++) {
			x = layers[i].Forward(x);
		}
		return x;
	}

	//score the overall network perfomance, used to tell user how good the network is.
	float Network::loss(const std::vector<float>& p, const std::vector<float>& y, LossTypes lossType) {
		float loss = 0.0f;
		for (size_t i = 0; i < p.size(); i++) {

			switch (lossType) {
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
		if (lossType == LossTypes::MSE) {
			return loss / p.size();
		}
		return loss;
	}

	//score the neural network
	float Network::lossDerivative(float p, float y, LossTypes lossType) {
		switch (lossType) {
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

	void Network::backpropagate(const std::vector<float>& targets) {
		for (int i = (int)layers.size() - 1; i >= 0; i--) {
			bool isOutput = (i == layers.size() - 1);

			if (isOutput) {
				layers[i].backpropagation(targets, {}, true);
			} else {
				layers[i].backpropagation(layers[i + 1].getDelta(), layers[i + 1].getWeights(), false);
			}
		}
	}

	void Network::updateNetwork(const std::vector<float>& inputs, float learningRate) {
		for (size_t i = 0; i < layers.size(); i++) {
			if (i == 0) {
				layers[i].updateWeights(inputs, learningRate);
			}else {
				layers[i].updateWeights(layers[i - 1].getResult(), learningRate);
			}
		}
	}

	float Network::epoch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, float learningRate, LossTypes losstype) {
		float total_loss = 0.0f;
		assert(inputs.size() == targets.size());
		for (size_t i = 0; i < inputs.size(); i++)
		{
			const std::vector<float>& x = inputs[i];
			const std::vector<float>& y = targets[i];

			std::vector<float> prediction = forward(x);
			total_loss += loss(prediction, y, LossTypes::crossEntropy);
			backpropagate(y);
			updateNetwork(x, learningRate);
		}
		return total_loss / inputs.size();
	}

	void Network::train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets,int epochs, float learningRate, LossTypes losstype) {
		std::cout << "Starting Training..." << std::endl;
		for (int i = 0; i < epochs; i++) {
			float avgLoss = epoch(inputs, targets, learningRate, losstype);
			if (i % 100 == 0) {
				std::cout.precision(8);
				std::cout << std::fixed;
				std::cout << "Epoch: " << i
					<< " | Loss: " << avgLoss << std::endl;
			}
			std::cout.unsetf(std::ios::fixed);
		}
	}

}