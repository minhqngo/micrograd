import numpy as np
from micrograd.engine import Value
from micrograd.optimizer import SGD, NesterovSGD

np.random.seed(42)


class PrintColor:
    PASS = '\033[32m'
    FAIL = '\033[31m'
    ENDC = '\033[0m'

DONE_TEST = "-------------------------------------------------------------"


def test_sgd():
    param = Value(np.array(10.0))
    optimizer = SGD([param], learning_rate=0.1)
    loss = param * param

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # w_1 = w_0 - lr * grad
    # grad = 2 * param = 2 * 10 = 20
    # w_1 = 10 - 0.1 * 20 = 8.0
    expected_param = 8.0

    assert np.allclose(param.data, expected_param), PrintColor.FAIL + f"SGD test failed: expected {expected_param}, got {param.data}" + PrintColor.ENDC
    print(PrintColor.PASS + "SGD test passed!" + PrintColor.ENDC)
    print(DONE_TEST)


def test_nesterov_sgd():
    param = Value(np.array(10.0))
    optimizer = NesterovSGD([param], learning_rate=0.1, momentum=0.9)
    loss = param * param

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # v_t = mu * v_{t-1} - lr * grad
    # w_t = w_{t-1} + v_t
    # v_0 = 0
    # grad = 2 * param = 2 * 10 = 20
    # v_1 = 0.9 * 0 - 0.1 * 20 = -2.0
    # w_1 = 10 + (-0.9 * 0 + (1 + 0.9) * -2.0) = 10 - 3.8 = 6.2
    expected_param = 6.2

    assert np.allclose(param.data, expected_param), PrintColor.FAIL + f"NesterovSGD test failed: expected {expected_param}, got {param.data}" + PrintColor.ENDC
    print(PrintColor.PASS + "NesterovSGD test passed!" + PrintColor.ENDC)
    print(DONE_TEST)


if __name__ == '__main__':
    test_sgd()
    test_nesterov_sgd()
