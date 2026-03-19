<div align="center">

# MiniNeuron

Small neural network library written from scratch in c++

![Language](https://img.shields.io/badge/language-C%2B%2B-blue)
![Version](https://img.shields.io/badge/version-1.0.0-orange)

</div>

## Features
- Dense (fully connected) layers
- Activations: Sigmoid, ReLU, Softmax
- Loss functions: MSE, Cross Entropy
- Initializers: Xavier, He
- Simple training API

## Quick example for XOR
```cpp
MiniNeuron::Network net;

//structure of layers 2 -> 4 -> 1
net.add(MiniNeuron::Layer(4, 2, ActivationType::Sigmoid, InitializerType::Xavier));
net.add(MiniNeuron::Layer(1, 4, ActivationType::Sigmoid, InitializerType::Xavier));

net.initLayers();

net.train(inputs, targets, 500, 5.0f, LossTypes::crossEntropy);

std::vector<float> prediction = net.forward(input);
```

## Benchmarks
V 1.0.0
| Demo | Network | Epochs | Accuracy | Time | CPU |
|------|---------|--------|----------|------|-----|
| XOR | 2→4→1 | 500 | Loss: 0.01 | ~1s | AMD Ryzen 5 5600 |
| MNIST (Sigmoid) | 784→128→10 | 1 | 95.66% | ~5min | AMD Ryzen 5 5600 |
| MNIST (ReLU)* | 784→256→128→10 | 3 | 96.98% | ~37min | AMD Ryzen 5 5600 |

## Roadmap
- [ ] Batch training
- [ ] Multi-core support
- [ ] CUDA support
- [ ] More activations (Tanh, Leaky ReLU)
- [ ] Unit tests
- [ ] Full documentation
