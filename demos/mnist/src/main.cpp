#include "Layer.h"
#include "network.h"
#include "activation_types.h"
#include "loss_types.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

//struct to store mnsit data
struct MNISTData {
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;
};

//for loadign the mnist data from csv file
MNISTData loadMnist(const std::string& path, size_t size) {
    std::ifstream file(path);

    MNISTData data;

    if (!file.is_open()) {
        std::cout << "Failed to open file" << std::endl;
        return data;
    }

    data.inputs.reserve(size);
    data.targets.reserve(size);

    std::string line;
    std::getline(file, line);
    size_t linesLoaded = 1;
    while (std::getline(file, line)) {
        std::vector<float> img;
        img.reserve(784);
        std::vector<float> target(10, 0.0f);

        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');

        target[stoi(cell)] = 1.0f;

        while (std::getline(ss, cell, ',')) {
            img.push_back(std::stof(cell) / 255.0f);
        }
        data.inputs.push_back(img);
        data.targets.push_back(target);
        std::cout << "\rProgress: " << linesLoaded << std::flush;
        linesLoaded++;
    }
    std::cout << std::endl;

    file.close();
    return data;
}

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<int> loadLabels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Failed to open: " << path << std::endl;
        return {};
    }
    int magicNumber = 0;
    file.read((char*)&magicNumber, 4);
    magicNumber = reverseInt(magicNumber);

    int numLabels = 0;
    file.read((char*)&numLabels, 4);
    numLabels = reverseInt(numLabels);

    std::vector<int> labels(numLabels);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        file.read((char*)&label, 1);
        labels[i] = (int)label;
    }

    return labels;
}

std::vector<std::vector<float>> loadImages(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Failed to open: " << path << std::endl;
        return {};
    }
    int magicNumber = 0;
    file.read((char*)&magicNumber, 4);
    magicNumber = reverseInt(magicNumber);

    int numImgs = 0;
    file.read((char*)&numImgs, 4);
    numImgs = reverseInt(numImgs);

    int rows = 0, cols = 0;
    file.read((char*)&rows, 4);
    file.read((char*)&cols, 4);
    rows = reverseInt(rows);
    cols = reverseInt(cols);

    std::vector<std::vector<float>> imgs(numImgs);
    for (int i = 0; i < numImgs; i++) {
        std::vector<float> img;
        img.reserve(784);
        for (int j = 0; j < 784; j++) {
            unsigned char pixel;
            file.read((char*)&pixel, 1);
            img.push_back(pixel / 255.0f);
        }
        imgs[i] = img;
    }
    return imgs;
}

MNISTData loadMnistV2(const std::string& imgPath, const std::string& labelPath) {
    MNISTData data;
    data.inputs = loadImages(imgPath);
    std::vector<int> labels = loadLabels(labelPath);
    for (int i = 0; i < labels.size(); i++) {
        std::vector<float> target(10, 0.0f);
        target[labels[i]] = 1.0f;
        data.targets.push_back(target);
    }
    return data;
}


int main() {
    //make the neural network object
    MiniNeuron::Network net;

    //structure of layers 784 -> 128 -> 10
    net.add(MiniNeuron::Layer(128, 784, ActivationType::Sigmoid, InitializerType::Xavier));
    net.add(MiniNeuron::Layer(10, 128, ActivationType::Softmax, InitializerType::Xavier));

    //init all layer weights, bias and other stuff
    net.initLayers();

    //define learning data for mnist
    MNISTData training = loadMnistV2("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    MNISTData testing = loadMnistV2("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
    std::cout << "Training samples: " << training.inputs.size() << std::endl;
    std::cout << "Testing samples: " << testing.inputs.size() << std::endl;

    float learningRate = 0.05f;

    //train function, 1 epoch.
    net.train(training.inputs, training.targets, 1, learningRate);

    
    //Global Accuracy Calculation
    std::cout << "\nCalculating Global Accuracy..." << std::endl;

    int correctPredictions = 0;
    size_t totalTestSamples = testing.inputs.size();

    for (size_t i = 0; i < totalTestSamples; i++) {
        std::vector<float> res = net.forward(testing.inputs[i]);

        int prediction = 0;
        float maxProb = res[0];
        for (int j = 1; j < 10; j++) {
            if (res[j] > maxProb) {
                maxProb = res[j];
                prediction = j;
            }
        }

        int actual = 0;
        for (int j = 0; j < 10; j++) {
            if (testing.targets[i][j] == 1.0f) {
                actual = j;
                break;
            }
        }

        if (prediction == actual) {
            correctPredictions++;
        }

        if (i % 500 == 0) {
            std::cout << "\rEvaluating: " << i << "/" << totalTestSamples << std::flush;
        }
    }

    float accuracy = (static_cast<float>(correctPredictions) / totalTestSamples) * 100.0f;

    std::cout << "\rEvaluation Complete!                          " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << " Final Test Accuracy: " << accuracy << "%" << std::endl;
    std::cout << " Total Correct: " << correctPredictions << " / " << totalTestSamples << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}