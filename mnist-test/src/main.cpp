#include "Layer.h"
#include <iostream>
#include <vector>

int main() {

	MiniNeuron::Layer testLayer(4, 3);
	std::vector<float> input = { 1.0, 1.0, 1.0 };
	testLayer.initWeights();
	testLayer.printWeights();
	std::vector<float> ans = testLayer.Forward(input);
	std::cout << "results:" << std::endl;
	for (int i = 0; i < ans.size(); i++) {
		std::cout << ans[i] << ", ";
	}
}