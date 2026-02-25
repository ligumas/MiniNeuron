#include "Layer.h"
#include <iostream>
#include <random>

namespace MiniNeuron {

	Layer::Layer(size_t NeuronCount, size_t InputCount) {
		this->NeuronCount = NeuronCount;
		this->InputCount =  InputCount;
	}

	void Layer::initWeights() {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(-0.05f, 0.05f);

		float random_num = dis(gen);

		weights.resize(NeuronCount, std::vector<float>(InputCount, 0.0f));
		bias.resize(NeuronCount, 1);
		             
		for(size_t i = 0; i < weights.size(); i++) {
			random_num = dis(gen);
			bias[i] = random_num;
			for (size_t j = 0; j < weights[i].size(); j++)
			{
				random_num = dis(gen);
				weights[i][j] = random_num;
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

}