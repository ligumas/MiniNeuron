#pragma once

#include "Layer.h"
#include "loss_types.h"
#include <vector>

namespace MiniNeuron {

	class Network {
	public:

		//core functions
		void add(Layer&& x);
		void initLayers();
		std::vector<float> forward(const std::vector<float>& inputs);
		void backpropagate(const std::vector<float>& targets);
		void updateNetwork(const std::vector<float>& inputs, float learningRate);
		float epoch(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, float learningRate);
		void train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, int epochs, float learningRate);

		//loss
		float loss(const std::vector<float>& p, const std::vector<float>& y, LossTypes losstype);
		float lossDerivative(float p, float y, LossTypes losstype);

		//debug
		void printNetwork();

	private:
		std::vector<Layer> layers;
	};

}