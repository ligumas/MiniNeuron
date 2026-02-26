#include "network.h"
#include "Layer.h"
#include <vector>
#include <iostream>

namespace MiniNeuron {

	void network::add(Layer&& x) {
		layers.push_back(x);
		std::cout << "added layer to network" << std::endl;
	}

	void network::initlayers() {
		for (size_t i = 0; i < layers.size(); i++) {
			layers[i].initWeights();
		}
	}

	void network::printNetwork() {
		for (size_t i = 0; i < layers.size(); i++) {
			std::cout << "Layer(" << i << ")" << " neurons(" << layers[i].NeuronCount << ")" << std::endl;
		}
	}

}