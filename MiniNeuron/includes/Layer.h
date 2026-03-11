#pragma once
#include <vector>
#include "activation_types.h"

namespace MiniNeuron {

	class Layer
	{
	public:
		Layer(size_t NeuronCount, size_t InputCount, ActivationType);

		void initWeights();
		void printWeights();
		float calculate_activation_derivative(float);

		std::vector<float> Forward(std::vector<float>& Input);
		void backpropagation(const std::vector<float>& prev_delta, const std::vector<std::vector<float>>& prev_weights, bool isOutput);
		void updateWeights(const std::vector<float>& inputs, float learningRate);

		//result with activation
		std::vector<float> result;
		//raw result
		std::vector<float> z_result;
		std::vector<float> Delta;

		size_t NeuronCount;
		size_t InputCount;

		std::vector<std::vector<float>> weights;
		std::vector<float> bias;

		ActivationType activation = ActivationType::Linear;
	
		
	};


}