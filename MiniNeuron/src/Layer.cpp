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
		             
		for(size_t i = 0; i < weights.size(); i++) {
			for (size_t j = 0; j < weights[i].size(); j++)
			{
				random_num = dis(gen);
				weights[i][j] = random_num;
			}
		}
		std::cout << "weights initialized" << std::endl;
	}

	void Layer::printWeights() {
		for (size_t i = 0; i < weights.size(); i++)
		{
			for (size_t j = 0; j < weights[i].size(); j++) {
				std::cout << weights[i][j] << ", ";
			}
			std::cout << std::endl;
		}
		std::cout << "printed weights" << std::endl;
	}

}