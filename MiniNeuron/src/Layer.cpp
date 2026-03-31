#include "Layer.h"
#include "initializers.h"
#include "activation_types.h"
#include "Matrix.h"
#include <iostream>
#include <cmath>

namespace MiniNeuron {

	Layer::Layer(size_t neuronCount, size_t inputCount, ActivationType activation, InitializerType initializer) : m_weights(neuronCount, inputCount) {
		this->m_neuronCount = neuronCount;
		this->m_inputCount = inputCount;
		this->activation = activation;
		this->initializer = initializer;	
	}

	void Layer::initSizes() {
		m_biases.resize(m_neuronCount, 0);
		m_delta.resize(m_neuronCount, 0.0f);
		m_zResult.resize(m_neuronCount, 0.0f);
		m_result.resize(m_neuronCount, 0.0f);
	}

	void Layer::initLayer() {

		initSizes();
		
		switch (initializer) {
		case InitializerType::HeInit:
		{
			#pragma omp parallel for
			for (int i = 0; i < m_weights.cols * m_weights.rows; i++) {
				m_weights.data[i] = HeInit(m_inputCount);
			}
			break;
		}
		case InitializerType::Xavier:
		{
			#pragma omp parallel for
			for (int i = 0; i < m_weights.cols * m_weights.rows; i++) {
					m_weights.data[i] = Xavier(m_inputCount, m_neuronCount);
			}
			break;
		}
		}
		std::cout << "weights initialized" << std::endl; // debug print
	}

	std::vector<float> Layer::Forward(const std::vector<float>& input) {
		std::fill(m_result.begin(), m_result.end(), 0.0f);

		if (m_weights.cols != input.size()) {
			std::cout << "Invalid input in matrix multiplication " << std::endl;
			std::invalid_argument;
		}
		#pragma omp parallel for
		for (int i = 0; i < m_neuronCount; i++) {
			for (int j = 0; j < m_inputCount; j++) {
				m_result[i] += input[j] * m_weights(i, j);
			}
			m_result[i] += m_biases[i];
		}

		//copy result before adding activation
		m_zResult = m_result;

		//switch case to choose the activasion method
		switch (activation) {
		case ActivationType::ReLU:
		{
			for (int i = 0; i < m_neuronCount; i++) {
				m_result[i] = std::max(0.0f, m_result[i]);
			}
			break;
		}
		case ActivationType::Softmax:
		{
			float max_val = m_result[0];
			for (size_t i = 1; i < m_neuronCount; i++) {
				if (m_result[i] > max_val) max_val = m_result[i];
			}
			float sum = 0.0f;
			for (size_t i = 0; i < m_neuronCount; i++) {
				m_result[i] = std::exp(m_result[i] - max_val);
				sum += m_result[i];
			}
			if (sum == 0.0f) {
				throw std::invalid_argument("Softmax sum is zero");
			}
			for (int i = 0; i < m_neuronCount; i++) {
				m_result[i] /= sum;
			}
			break;
		}
		case ActivationType::Sigmoid:
		{
			for (int i = 0; i < m_neuronCount; i++) {
				m_result[i] = 1.0f / (1.0f + std::exp(-m_result[i]));
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

	void Layer::backpropagation(const std::vector<float>& prev_delta, const Matrix& previous_weights, bool isOutput) {

#pragma omp parallel for
		for (int i = 0; i < m_neuronCount; i++) {
			float error = 0.0f;

			if (isOutput) {
				error = (m_result[i] - prev_delta[i]);
			}
			else {
				for (int j = 0; j < prev_delta.size(); j++) {
					error += prev_delta[j] * previous_weights(j, i);
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
				m_weights(i, j) -= change;
			}
			m_biases[i] -= learningRate * m_delta[i];
		}
	}

}