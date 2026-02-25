#include "Layer.h"
#include <iostream>

int main() {

	MiniNeuron::Layer testLayer(4, 3);

	testLayer.initWeights();
	testLayer.printWeights();
}