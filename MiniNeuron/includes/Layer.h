#pragma once
#include <vector>
#include "activation_types.h"

namespace MiniNeuron {

	class Layer
	{
	public:
		Layer(size_t InputCount, size_t NeuronCount, ActivationType activation);

		void initWeights();
		void printWeights();

		std::vector<float> Forward(std::vector<float>& Input);
		std::vector<float> result;

		size_t NeuronCount;
		size_t InputCount;

	private:

		ActivationType activation = ActivationType::Linear;

		std::vector<std::vector<float>> weights;
		std::vector<float> bias;
	};


}