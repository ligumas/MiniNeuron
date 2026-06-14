<div align="center">

# MiniNeuron

train feedforward neural networks in C++ — only dependency is optional OpenMP

![Language](https://img.shields.io/badge/language-C%2B%2B-blue)
![Version](https://img.shields.io/badge/version-1.1.0-orange)

</div>

**build:**

```
git clone https://github.com/ligumas/MiniNeuron
cd MiniNeuron
cmake -B build && cmake --build build
```

## Quick example (XOR)

```cpp
#include "MiniNeuron/includes/network.h"

MiniNeuron::Network net;

// structure: 2 → 4 → 1
net.add(MiniNeuron::Layer(4, 2, ActivationType::Sigmoid, InitializerType::Xavier));
net.add(MiniNeuron::Layer(1, 4, ActivationType::Sigmoid, InitializerType::Xavier));
net.initLayers();

// inputs and targets are std::vector<std::vector<float>>
net.train(inputs, targets, 1, 500, 5.0f, LossTypes::crossEntropy);

std::vector<float> prediction = net.forward(input);

// save to a .mn file
net.saveModel("xor-model.mn");
```

## Features

- Dense (fully connected) layers
- Activations: Sigmoid, ReLU, Tanh, LeakyReLU, Softmax, Linear
- Loss functions: MSE, Cross Entropy
- Optimizers: SGD, Adam
- Initializers: Xavier, He
- Multi-threading via OpenMP (optional)
- Model save/load (.mn format)

## Benchmarks

**v1.1.0 — with OpenMP**
| Demo | Network | Epochs | Accuracy | Time | CPU |
|------|---------|--------|----------|------|-----|
| XOR | 2→4→1 | 500 | Loss: 0.0133 | ~0s | AMD Ryzen 5 5600 |
| MNIST (Sigmoid) | 784→128→10 | 1 | 95.65% | ~1s | AMD Ryzen 5 5600 |
| MNIST (ReLU) | 784→256→128→10 | 3 | 97.36% | ~12s | AMD Ryzen 5 5600 |

**v1.0.0 — single-threaded**
| Demo | Network | Epochs | Accuracy | Time | CPU |
|------|---------|--------|----------|------|-----|
| XOR | 2→4→1 | 500 | Loss: 0.01 | ~1s | AMD Ryzen 5 5600 |
| MNIST (Sigmoid) | 784→128→10 | 1 | 95.66% | ~5min | AMD Ryzen 5 5600 |
| MNIST (ReLU) | 784→256→128→10 | 3 | 96.98% | ~37min | AMD Ryzen 5 5600 |

## License

MIT
