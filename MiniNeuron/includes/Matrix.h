#pragma once

#include <vector>

namespace MiniNeuron {
	struct Matrix {

		int rows, cols;
		std::vector<float> data;

		Matrix(int x, int y) : rows(x), cols(y), data(x* y, 0.0f) {}

		inline float& operator()(int x, int y) {
			return data[x * cols + y];
		}

		inline float operator()(int r, int c) const {
			return data[r * cols + c];
		}

		void clear() {
			std::fill(data.begin(), data.end(), 0.0f);
		}

	};
}