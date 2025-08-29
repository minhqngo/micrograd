import numpy as np
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP

np.random.seed(42)


class PrintColor:
    PASS = '\033[32m'
    FAIL = '\033[31m'
    ENDC = '\033[0m'
    
DONE_TEST = "-------------------------------------------------------------"


def test_neuron():
    nin = 5
    neuron = Neuron(nin)
    assert len(neuron.w.data) == nin, PrintColor.FAIL + "Neuron weight initialization failed!" + PrintColor.ENDC
    assert np.allclose(neuron.b.data, [0]), PrintColor.FAIL + "Neuron bias initialization failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "Neuron initialization OK!" + PrintColor.ENDC)

    x = Value(np.random.randn(nin))
    y = neuron(x)
    z = x.data @ neuron.w.data + neuron.b.data
    assert np.allclose(y.data, np.maximum(0, z)), PrintColor.FAIL + "Neuron forward pass (non-linear) failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "Neuron forward pass (non-linear) OK!" + PrintColor.ENDC)

    neuron_linear = Neuron(nin, nonlin=False)
    y_linear = neuron_linear(x)
    z_linear = x.data @ neuron_linear.w.data + neuron_linear.b.data
    assert np.allclose(y_linear.data, z_linear), PrintColor.FAIL + "Neuron forward pass (linear) failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "Neuron forward pass (linear) OK!" + PrintColor.ENDC)
    
    params = neuron.parameters()
    assert len(params) == 2, PrintColor.FAIL + "Neuron parameters() failed!" + PrintColor.ENDC
    assert params[0] == neuron.w, PrintColor.FAIL + "Neuron parameters() weight failed!" + PrintColor.ENDC
    assert params[1] == neuron.b, PrintColor.FAIL + "Neuron parameters() bias failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "Neuron parameters() OK!" + PrintColor.ENDC)
    
    print(PrintColor.PASS + "All Neuron tests passed!" + PrintColor.ENDC)
    print(DONE_TEST)


def test_layer():
    nin, nout = 5, 10
    layer = Layer(nin, nout)
    assert layer.w.data.shape == (nin, nout), PrintColor.FAIL + "Layer weight initialization failed!" + PrintColor.ENDC
    assert layer.b.data.shape == (nout,), PrintColor.FAIL + "Layer bias initialization failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "Layer initialization OK!" + PrintColor.ENDC)

    x = Value(np.random.randn(1, nin))
    y = layer(x)
    z = x.data @ layer.w.data + layer.b.data
    assert np.allclose(y.data, np.maximum(0, z)), PrintColor.FAIL + "Layer forward pass (non-linear) failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "Layer forward pass (non-linear) OK!" + PrintColor.ENDC)

    layer_linear = Layer(nin, nout, nonlin=False)
    y_linear = layer_linear(x)
    z_linear = x.data @ layer_linear.w.data + layer_linear.b.data
    assert np.allclose(y_linear.data, z_linear), PrintColor.FAIL + "Layer forward pass (linear) failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "Layer forward pass (linear) OK!" + PrintColor.ENDC)

    params = layer.parameters()
    assert len(params) == 2, PrintColor.FAIL + "Layer parameters() failed!" + PrintColor.ENDC
    assert params[0] == layer.w, PrintColor.FAIL + "Layer parameters() weight failed!" + PrintColor.ENDC
    assert params[1] == layer.b, PrintColor.FAIL + "Layer parameters() bias failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "Layer parameters() OK!" + PrintColor.ENDC)

    print(PrintColor.PASS + "All Layer tests passed!" + PrintColor.ENDC)
    print(DONE_TEST)


def test_mlp():
    nin, nouts = 5, [10, 20, 1]
    mlp = MLP(nin, nouts)
    assert len(mlp.layers) == len(nouts), PrintColor.FAIL + "MLP initialization failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "MLP initialization OK!" + PrintColor.ENDC)

    x = Value(np.random.randn(1, nin))
    y = mlp(x)
    assert y.data.shape == (1, nouts[-1]), PrintColor.FAIL + "MLP forward pass failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "MLP forward pass OK!" + PrintColor.ENDC)

    params = mlp.parameters()
    expected_num_params = len(nouts) * 2 # 2 params (w, b) per layer
    assert len(params) == expected_num_params, PrintColor.FAIL + "MLP parameters() failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "MLP parameters() OK!" + PrintColor.ENDC)

    print(PrintColor.PASS + "All MLP tests passed!" + PrintColor.ENDC)
    print(DONE_TEST)


if __name__ == '__main__':
    test_neuron()
    test_layer()
    test_mlp()
