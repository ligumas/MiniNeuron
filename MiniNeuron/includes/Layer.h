#pragma once
#include <vector>

namespace MiniNeuron {

	class Layer
	{
	public:
		Layer(size_t InputCount, size_t NeuronCount);

		void initWeights();
		void printWeights();

		std::vector<float> Forward(std::vector<float>& Input);
		std::vector<float> result;

		size_t NeuronCount;
		size_t InputCount;

	private:
		std::vector<std::vector<float>> weights;
		std::vector<float> bias;
	};


}