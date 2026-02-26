#include "Layer.h"
#include "network.h"
#include "activation_types.h"
#include <iostream>
#include <vector>

int main() {

	/*
	MiniNeuron::Layer testLayer(4, 3);
	std::vector<float> input = { 1.0, 1.0, 1.0 };
	testLayer.initWeights();
	testLayer.printWeights();
	std::vector<float> ans = testLayer.Forward(input);
	std::cout << "results:" << std::endl;
	for (int i = 0; i < ans.size(); i++) {
		std::cout << ans[i] << ", ";
	}
	*/

	MiniNeuron::network test_network;
	test_network.add(MiniNeuron::Layer(784, 256, ActivationType::ReLU));
	test_network.add(MiniNeuron::Layer(256, 128, ActivationType::ReLU));
	test_network.add(MiniNeuron::Layer(128, 10, ActivationType::Softmax));

	test_network.initlayers();

	test_network.printNetwork();

	return 0;
}