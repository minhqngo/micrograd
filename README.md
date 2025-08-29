This is a tiny autograd engine based on Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) library. This library is an extension of the original one.

* Support vector-valued and matrix-valued objects.
* Implement CE Loss and SGD for training neural network models. Currently only support MLP (multilayer perceptron).

I used this library to build neural network for digit classification on MNIST dataset, which achieved an 92% accuracy. I also used this library to build neural network for sentiment classification on [UCI sentiment dataset](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), achieved an 81% accuracy.