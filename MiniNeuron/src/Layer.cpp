#include "Layer.h"
#include "initializers.h"
#include "activation_types.h"
#include <iostream>

namespace MiniNeuron {

	Layer::Layer(size_t NeuronCount, size_t InputCount, ActivationType activation) {
		this->NeuronCount = NeuronCount;
		this->InputCount =  InputCount;
		this->activation = activation;
	}

	void Layer::initWeights() {
		weights.resize(NeuronCount, std::vector<float>(InputCount, 0.0f));
		bias.resize(NeuronCount, 0);
		Delta.resize(NeuronCount, 0.0f);
		z_result.resize(NeuronCount, 0.0f);
		result.resize(NeuronCount, 0.0f);
		             
		for(size_t i = 0; i < weights.size(); i++) {
			for (size_t j = 0; j < weights[i].size(); j++)
			{
				weights[i][j] = HeInit(InputCount);
				//weights[i][j] = ((float)rand() / (float)RAND_MAX) * 0.2f - 0.1f;
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

		result.assign(NeuronCount, 0.0f);

		for (size_t i = 0; i < NeuronCount; i++) {
			for (size_t j = 0; j < InputCount; j++) {
				result[i] += input[j] * weights[i][j];
			}
			result[i] += bias[i];
		}

		//copy result before adding activation
		z_result = result;

		//switch case to choose the activasion method
		switch (activation) {
			case ActivationType::ReLU:
				for(size_t i = 0; i < NeuronCount; i++) {
					result[i] = std::max(0.0f, result[i]);
				}
				break;
			case ActivationType::Softmax:
			{
				float max_val = result[0];
				for (size_t i = 1; i < NeuronCount; i++) {
					if (result[i] > max_val) max_val = result[i];
				}
				float sum = 0.0f;
				for (size_t i = 0; i < NeuronCount; i++) {
					result[i] = std::exp(result[i] - max_val);
					sum += result[i];
				}
				if (sum == 0.0f) {
					throw std::invalid_argument("Softmax sum is zero");
				}

				for (size_t i = 0; i < NeuronCount; i++) {
					result[i] /= sum;
				}
				break;
			}
			default:
				//default to linear
				break;
		}
		return(result);
	}

	float Layer::calculate_activation_derivative(float val) {
		switch (activation) {
			case ActivationType::ReLU:
			{
				return (val > 0.0f) ? 1.0f : 0.0f;
			}
			case ActivationType::Softmax:
			{
				return 1.0f;
			}
			default:
				return 1.0f; // linear

		}
	}

	void Layer::backpropagation(const std::vector<float>& prev_delta, const std::vector<std::vector<float>>& prev_weights, bool isOutput) {
		for (int i = 0; i < NeuronCount; i++) {
			float error = 0.0f;

			if(isOutput) {
				error = (result[i] - prev_delta[i]);
			} else {
				for (int j = 0; j < prev_delta.size(); j++) {
					error += prev_delta[j] * prev_weights[j][i];
				}
			}
			
			this->Delta[i] = error * calculate_activation_derivative(z_result[i]);
		}
	}

	void Layer::updateWeights(const std::vector<float>& input, float learningRate) {
		for (size_t i = 0; i < NeuronCount; i++) {
			for (size_t j = 0; j < InputCount; j++) {
				float change = learningRate * Delta[i] * input[j];
				weights[i][j] -= change;
			}
			bias[i] -= learningRate * Delta[i];
		}
	}

}