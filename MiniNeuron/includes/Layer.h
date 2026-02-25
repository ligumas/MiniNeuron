#pragma once
#include <vector>

namespace MiniNeuron {

	class Layer
	{
	public:
		Layer(size_t NeuronCount, size_t InputCount);

		void initWeights();
		void printWeights();

		std::vector<float> Forward(std::vector<float>& Input);

	private:
		std::vector<std::vector<float>> weights;
		std::vector<float> bias;

		size_t NeuronCount;
		size_t InputCount;
	};


}