<div align="center">

# MiniNeuron

C++ neural network library with OpenMP multithreading

![Language](https://img.shields.io/badge/language-C%2B%2B17-blue)
![Version](https://img.shields.io/badge/version-1.1.0-orange)
![license](https://img.shields.io/badge/license-MIT-blue?style=flat-square)

</div>

## Features
- Dense (fully connected) layers
- Activations: Sigmoid, ReLU, Softmax, Tanh, Leaky ReLU
- Optimizers: SGD, Adam
- Loss functions: MSE, Cross Entropy
- Initializers: Xavier, He
- Simple training API
- Model save/load (.mn format)
- OpenMP multithreading

## Quick example

```cpp
MiniNeuron::Network net;

// 2 -> 4 -> 1
net.add(MiniNeuron::Layer(4, 2, ActivationType::Sigmoid, InitializerType::Xavier));
net.add(MiniNeuron::Layer(1, 4, ActivationType::Sigmoid, InitializerType::Xavier));

net.initLayers();

// inputs and targets are std::vector<std::vector<float>>
net.train(inputs, targets, 500, 5.0f, LossTypes::crossEntropy);

std::vector<float> prediction = net.forward(input);

// save to .mn file
net.saveModel("xor-model.mn");
```

## Benchmarks

**v1.1.0 — with OpenMP multithreading**
| Demo | Network | Epochs | Performance | Time | CPU |
|------|---------|--------|-------------|------|-----|
| XOR | 2->4->1 | 500 | Loss: 0.0133 | ~0s | AMD Ryzen 5 5600 |
| MNIST (Sigmoid) | 784->128->10 | 1 | 95.65% | ~1s | AMD Ryzen 5 5600 |
| MNIST (ReLU) | 784->256->128->10 | 3 | 97.36% | ~12s | AMD Ryzen 5 5600 |

**v1.0.0 — no multithreading**
| Demo | Network | Epochs | Performance | Time | CPU |
|------|---------|--------|-------------|------|-----|
| XOR | 2->4->1 | 500 | Loss: 0.01 | ~1s | AMD Ryzen 5 5600 |
| MNIST (Sigmoid) | 784->128->10 | 1 | 95.66% | ~5min | AMD Ryzen 5 5600 |
| MNIST (ReLU) | 784->256->128->10 | 3 | 96.98% | ~37min | AMD Ryzen 5 5600 |

## Roadmap
- [ ] Batch training
- [x] Multi-core support (OpenMP)
- [ ] CUDA support
- [x] Tanh, Leaky ReLU activations
- [x] Adam optimizer
- [ ] Unit tests
- [ ] Full documentation
