#pragma once
#include <vector>
#include "Matrix.h"
#include "activation_types.h"
#include "initializers.h"

namespace MiniNeuron {

	class Layer
	{
	public:
		Layer(size_t neuronCount, size_t inputCount, ActivationType, InitializerType); //constructor

		//weight initilization
		void initSizes();
		void initLayer(); 

		//Core functionality
		float calculateActivationDerivative(float);
		const std::vector<float>& Forward(const std::vector<float>& Input);
		void backpropagation(const std::vector<float>& previous_delta,const Matrix& previous_weights, const std::vector<float>& input, bool isOutput);
		void updateWeights(int batchSize, float learningRate);

		//get functions
		size_t getNeuronCount() const { return m_neuronCount; }
		size_t getInputCount() const { return m_inputCount; }
		const Matrix& getWeights() const { return m_weights; };
		const std::vector<float> getBiases() const { return m_biases; };
		const std::vector<float>& getDelta() const { return m_delta; };
		const std::vector<float>& getResult() const { return m_result; };
		ActivationType getActivation() { return activation; }
		InitializerType getInitializer() { return initializer; }

		//set functions
		void setWeights(const Matrix& weights) { m_weights = weights; }
		void setBiases(const std::vector<float>& biases) { m_biases = biases; }
	
	private:
		int m_neuronCount;
		int m_inputCount;

		InitializerType initializer = InitializerType::Xavier;
		ActivationType activation = ActivationType::Linear;

		Matrix m_weights;
		std::vector<float> m_biases;
		std::vector<float> m_result; //final result after activation
		std::vector<float> m_zResult; //pre-activation result
		std::vector<float> m_delta;

		//gradients
		Matrix m_weightGradients;
		std::vector<float> m_biasGradients;
	};


}