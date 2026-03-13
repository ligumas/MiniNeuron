#pragma once

#include "Layer.h"
#include "loss_types.h"

#include <vector>

namespace MiniNeuron {

	class network {
	public:

		void add(MiniNeuron::Layer&& x);
		void initlayers();
		std::vector<float> forward(std::vector<float> inputs);
		void printNetwork();
		std::vector<MiniNeuron::Layer> layers;
		float loss(const std::vector<float>& p, const std::vector<float>& y, LossTypes losstype);
		float loss_derivative(float p, float y, LossTypes losstype);
		void backpropagate(const std::vector<float>& targets);
		void update_network(const std::vector<float>& og_inputs, float learningRate);
		float epoch(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& target, float learning_rate);
		void train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, int epochs, float learning_rate);

	private:
		
	};

}