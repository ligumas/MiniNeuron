#pragma once

#include "Layer.h"

#include <vector>

namespace MiniNeuron {

	class network {
	public:

		void add(MiniNeuron::Layer&& x);
		void initlayers();
		std::vector<float> forward();
		void printNetwork();

	private:
		std::vector<MiniNeuron::Layer> layers;
	};

}