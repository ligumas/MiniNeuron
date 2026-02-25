#include "initializers.h";
#include <cmath>
#include <random>

float HeInit(size_t n) {
	float stddev = std::sqrt(2.0f / static_cast<float>(n));

	static std::default_random_engine gen(std::random_device{}());

	std::normal_distribution<float> dis(0.0, stddev);

	return dis(gen);
}