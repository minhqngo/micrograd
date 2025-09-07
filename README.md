This is a tiny autograd engine, with a tiny neural network library built on top of it. This project is based on Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) project. It is an extension to Andrej's project, which supports vector-valued and matrix-valued objects, unlike the original which only supports scalars.

The neural networks library built on top of this autograd engine has these features:
* Supports loss functions: CE loss, BCE loss, MSE loss.
* Support optimizers: SGD, SGD with momentum.
* Currently support downloading and loading MNIST dataset and UCI sentiment dataset.

I used this library to build neural network for digit classification on MNIST dataset, which achieved an 92% accuracy. I also used this library to build neural network for sentiment classification on [UCI sentiment dataset](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), achieved an 81% accuracy.