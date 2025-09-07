This is a tiny autograd engine based on Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) library. This library is an extension of the original one.

* Support vector-valued and matrix-valued objects.
* Supports loss functions: CE loss, BCE loss, MSE loss.
* Support optimizers: SGD, SGD with momentum.
* Currently support downloading and loading MNIST dataset and UCI sentiment dataset.

I used this library to build neural network for digit classification on MNIST dataset, which achieved an 92% accuracy. I also used this library to build neural network for sentiment classification on [UCI sentiment dataset](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), achieved an 81% accuracy.