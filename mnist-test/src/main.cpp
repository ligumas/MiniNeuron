#include "Layer.h"
#include "network.h"
#include "activation_types.h"
#include "loss_types.h"
#include <iostream>
#include <vector>

int main() {
    MiniNeuron::network net;

    // 2 Inputs -> 4 Hidden Neurons -> 1 Output Neuron
    net.add(MiniNeuron::Layer(4, 2, ActivationType::ReLU));
    net.add(MiniNeuron::Layer(1, 4, ActivationType::Linear)); // Output layer (size 1)

    net.initlayers();

    // XOR Data: {Input1, Input2} -> {Target}
    std::vector<std::vector<float>> inputs = { {0,0}, {0,1}, {1,0}, {1,1} };
    std::vector<std::vector<float>> targets = { {0}, {1}, {1}, {0} };

    float learning_rate = 0.0005f;

    net.train(inputs, targets, 50000, learning_rate);

    std::cout << "\nFinal Testing:" << std::endl;
    for (auto& test_in : inputs) {
        std::vector<float> res = net.forward(test_in);
        std::cout << "In: " << test_in[0] << "," << test_in[1]
            << " Predicted: " << std::round(res[0]) << std::endl;
    }

    return 0;
}