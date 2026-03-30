#pragma once
#include <vector>
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

		//debug function
		void printWeights();

		//Core functionality
		float calculateActivationDerivative(float);
	    std::vector<float> Forward(const std::vector<float>& Input);
		void backpropagation(const std::vector<float>& previous_delta, const std::vector<std::vector<float>>& previous_weights, bool isOutput);
		void updateWeights(const std::vector<float>& inputs, float learningRate);

		//get functions
		size_t getNeuronCount() const { return m_neuronCount; }
		size_t getInputCount() const { return m_inputCount; }
		const std::vector<std::vector<float>>& getWeights() const { return m_weights; };
		const std::vector<float> getBiases() const { return m_biases; };
		const std::vector<float>& getDelta() const { return m_delta; };
		const std::vector<float>& getResult() const { return m_result; };
		ActivationType getActivation() { return activation; }
		InitializerType getInitializer() { return initializer; }

		//set functions
		void setWeights(const std::vector<std::vector<float>>& weights) { m_weights = weights; }
		void setBiases(const std::vector<float>& biases) { m_biases = biases; }
	
	private:
		size_t m_neuronCount;
		size_t m_inputCount;

		InitializerType initializer = InitializerType::Xavier;
		ActivationType activation = ActivationType::Linear;

		std::vector<std::vector<float>> m_weights;
		std::vector<float> m_biases;
		std::vector<float> m_result; //final result after activation
		std::vector<float> m_zResult; //pre-activation result
		std::vector<float> m_delta;
	};


}