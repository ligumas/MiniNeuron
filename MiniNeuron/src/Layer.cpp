#include "Layer.h"
#include "initializers.h"
#include "activation_types.h"
#include <iostream>
#include <cmath>

namespace MiniNeuron {

	Layer::Layer(size_t neuronCount, size_t inputCount, ActivationType activation, InitializerType initializer) {
		this->m_neuronCount = neuronCount;
		this->m_inputCount = inputCount;
		this->activation = activation;
		this->initializer = initializer;
	}

	void Layer::initLayer() {
		//size all vectors for better and safer use.
		m_weights.resize(m_neuronCount, std::vector<float>(m_inputCount, 0.0f));
		m_biases.resize(m_neuronCount, 0);
		m_delta.resize(m_neuronCount, 0.0f);
		m_zResult.resize(m_neuronCount, 0.0f);
		m_result.resize(m_neuronCount, 0.0f);

		switch (initializer) {
		case InitializerType::HeInit:
		{
			#pragma omp parallel for
			for (int i = 0; i < m_neuronCount; i++) {
				for (size_t j = 0; j < m_weights[i].size(); j++)
				{
					m_weights[i][j] = HeInit(m_inputCount);
				}
			}
			break;
		}
		case InitializerType::Xavier:
		{
			#pragma omp parallel for
			for (int i = 0; i < m_neuronCount; i++) {
				for (size_t j = 0; j < m_weights[i].size(); j++)
				{
					m_weights[i][j] = Xavier(m_inputCount, m_neuronCount);
				}
			}
			break;
		}
		}
		std::cout << "weights initialized" << std::endl; // debug print
	}


	//debug function
	void Layer::printWeights() {
		std::cout << "Weights:" << "                                 bias:" << std::endl;
		for (size_t i = 0; i < m_neuronCount; i++)
		{
			for (size_t j = 0; j < m_weights[i].size(); j++) {
				std::cout << m_weights[i][j] << ", ";
			}

			std::cout << "       " << m_biases[i];

			std::cout << std::endl;
		}
		std::cout << "printed weights" << std::endl;
	}


	std::vector<float> Layer::Forward(const std::vector<float>& input) {

		if (m_weights[0].size() != input.size()) {
			std::cout << "Invalid input in matrix multiplication " << std::endl;
		}

		#pragma omp parallel for
		for (int i = 0; i < (int)m_neuronCount; i++) {
			float sum = 0.0f;
			for (size_t j = 0; j < m_inputCount; j++) {
				sum += input[j] * m_weights[i][j];
			}
			m_result[i] = sum + m_biases[i];
		}

		//copy result before adding activation
		m_zResult = m_result;

		//switch case to choose the activasion method
		switch (activation) {
		case ActivationType::ReLU:
		{
			#pragma omp parallel for
			for (int i = 0; i < m_neuronCount; i++) {
				m_result[i] = std::max(0.0f, m_result[i]);
			}
			break;
		}
		case ActivationType::Softmax:
		{
			float max_val = *std::max_element(m_result.begin(), m_result.end());
			float sum = 0.0f;
			for (auto& v : m_result) {
				v = std::exp(v - max_val);
				sum += v;
			}

			if (sum == 0.0f) {
				throw std::invalid_argument("Softmax sum is zero");
			}

			#pragma omp parallel for
			for (int i = 0; i < (int)m_neuronCount; i++) {
				m_result[i] /= sum;
			}
			break;
		}
		case ActivationType::Sigmoid:
		{
			#pragma omp parallel for
			for (int i = 0; i < m_neuronCount; i++) {
				m_result[i] = 1.0f / (1.0f + std::expf(-m_result[i]));
			}
			break;
		}
		default:
			//default to linear
			break;
		}
		return(m_result);
	}

	float Layer::calculateActivationDerivative(float val) {
		switch (activation) {
		case ActivationType::ReLU:
		{
			return (val > 0.0f) ? 1.0f : 0.0f;
		}
		case ActivationType::Softmax:
		{
			return 1.0f;
		}
		case ActivationType::Sigmoid:
		{
			float s = 1.0f / (1.0f + std::expf(-val));
			return s * (1.0f - s);
		}
		default:
			return 1.0f; // linear
		}
	}

	void Layer::backpropagation(const std::vector<float>& prev_delta, const std::vector<std::vector<float>>& prev_weights, bool isOutput) {
		#pragma omp parallel for
		for (int i = 0; i < m_neuronCount; i++) {
			float error = 0.0f;

			if (isOutput) {
				error = (m_result[i] - prev_delta[i]);
			}
			else {
				for (int j = 0; j < prev_delta.size(); j++) {
					error += prev_delta[j] * prev_weights[j][i];
				}
			}

			this->m_delta[i] = error * calculateActivationDerivative(m_zResult[i]);
		}
	}

	void Layer::updateWeights(const std::vector<float>& input, float learningRate) {
		#pragma omp parallel for
		for (int i = 0; i < m_neuronCount; i++) {
			for (int j = 0; j < m_inputCount; j++) {
				float change = learningRate * m_delta[i] * input[j];
				m_weights[i][j] -= change;
			}
			m_biases[i] -= learningRate * m_delta[i];
		}
	}

}