import numpy as np
from micrograd.engine import Value
from micrograd.loss import MSELoss, BinaryCrossEntropyLoss, CrossEntropyLoss

np.random.seed(42)


class PrintColor:
    PASS = '\033[32m'
    FAIL = '\033[31m'
    ENDC = '\033[0m'
    
DONE_TEST = "-------------------------------------------------------------"


def test_mse_loss():
    y_pred = Value(np.array([1, 2, 3]))
    y_true = Value(np.array([1.5, 2.5, 3.5]))

    mse_loss = MSELoss()
    loss = mse_loss(y_pred, y_true)

    expected_loss = np.mean((y_pred.data - y_true.data) ** 2)
    assert np.allclose(loss.data, expected_loss), PrintColor.FAIL + "MSELoss forward pass failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "MSELoss forward pass OK!" + PrintColor.ENDC)

    loss.backward()

    expected_grad = 2 * (y_pred.data - y_true.data) / y_true.data.size
    assert np.allclose(y_pred.grad, expected_grad), PrintColor.FAIL + "MSELoss backward pass failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "MSELoss backward pass OK!" + PrintColor.ENDC)

    print(PrintColor.PASS + "All MSELoss tests passed!" + PrintColor.ENDC)
    print(DONE_TEST)
    
    
def test_binary_cross_entropy_loss():
    y_pred = Value(np.array([0.2, 0.8, 0.6]))
    y_true = Value(np.array([0, 1, 1]))

    bce_loss = BinaryCrossEntropyLoss()
    loss = bce_loss(y_pred, y_true)

    probs = 1 / (1 + np.exp(-y_pred.data))
    expected_loss = -np.mean(y_true.data * np.log(probs) + (1 - y_true.data) * np.log(1 - probs))
    assert np.allclose(loss.data, expected_loss), PrintColor.FAIL + "BinaryCrossEntropyLoss forward pass failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "BinaryCrossEntropyLoss forward pass OK!" + PrintColor.ENDC)

    loss.backward()

    expected_grad = (probs - y_true.data) / y_true.data.size
    assert np.allclose(y_pred.grad, expected_grad), PrintColor.FAIL + "BinaryCrossEntropyLoss backward pass failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "BinaryCrossEntropyLoss backward pass OK!" + PrintColor.ENDC)

    print(PrintColor.PASS + "All BinaryCrossEntropyLoss tests passed!" + PrintColor.ENDC)
    print(DONE_TEST)
    
    
def test_cross_entropy_loss():
    logits = Value(np.array([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]]))
    true_labels = np.array([0, 1])

    ce_loss = CrossEntropyLoss()
    loss = ce_loss(logits, true_labels)

    probs = np.exp(logits.data) / np.sum(np.exp(logits.data), axis=1, keepdims=True)
    n_classes = probs.shape[1]
    true_labels_oh = np.eye(n_classes)[true_labels]
    expected_loss = -np.sum(true_labels_oh * np.log(probs)) / probs.shape[0]

    assert np.allclose(loss.data, expected_loss), PrintColor.FAIL + "CrossEntropyLoss forward pass failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "CrossEntropyLoss forward pass OK!" + PrintColor.ENDC)

    loss.backward()

    grad = (probs - true_labels_oh) / probs.shape[0]
    assert np.allclose(logits.grad, grad), PrintColor.FAIL + "CrossEntropyLoss backward pass failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "CrossEntropyLoss backward pass OK!" + PrintColor.ENDC)

    print(PrintColor.PASS + "All CrossEntropyLoss tests passed!" + PrintColor.ENDC)
    print(DONE_TEST)


if __name__ == '__main__':
    test_mse_loss()
    test_binary_cross_entropy_loss()
    test_cross_entropy_loss()
