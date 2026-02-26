#include "Layer.h"
#include "network.h"
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
	test_network.add(MiniNeuron::Layer(764, 400));
	test_network.add(MiniNeuron::Layer(400, 300));
	test_network.add(MiniNeuron::Layer(300, 10));

	test_network.initlayers();

	test_network.printNetwork();

	return 0;
}