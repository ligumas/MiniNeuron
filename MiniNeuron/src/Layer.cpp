#include "Layer.h"
#include "initializers.h"
#include <iostream>

namespace MiniNeuron {

	Layer::Layer(size_t NeuronCount, size_t InputCount) {
		this->NeuronCount = NeuronCount;
		this->InputCount =  InputCount;
	}

	void Layer::initWeights() {
		weights.resize(NeuronCount, std::vector<float>(InputCount, 0.0f));
		bias.resize(NeuronCount, 0);
		             
		for(size_t i = 0; i < weights.size(); i++) {
			for (size_t j = 0; j < weights[i].size(); j++)
			{
				weights[i][j] = HeInit(InputCount);
			}
		}
		std::cout << "weights initialized" << std::endl;
	}

	void Layer::printWeights() {
		std::cout << "Weights:" << "                                 bias:" << std::endl;
		for (size_t i = 0; i < weights.size(); i++)
		{
			for (size_t j = 0; j < weights[i].size(); j++) {
				std::cout << weights[i][j] << ", ";
			}

			std::cout << "       " << bias[i];

			std::cout << std::endl;
		}
		std::cout << "printed weights" << std::endl;
	}

	std::vector<float> Layer::Forward(std::vector<float>& input) {

		if (weights[0].size() != input.size()) {
			std::cout << "Invalid input in matrix multiplication " << std::endl;
			std::invalid_argument;
		}

		result.resize(NeuronCount, 0.0f);

		for (size_t i = 0; i < NeuronCount; i++) {
			for (size_t j = 0; j < InputCount; j++) {
				result[i] += input[j] * weights[i][j];
			}
			//linear output
			result[i] += bias[i];
			//ReLU move to own system later
			result[i] = std::max(0.0f, result[i]);
		}

		return(result);

	}
}