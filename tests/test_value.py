import numpy as np
from micrograd.engine import Value

np.random.seed(42)


class PrintColor:
    PASS = '\033[32m'
    FAIL = '\033[31m'
    ENDC = '\033[0m'
    
DONE_TEST = "-------------------------------------------------------------"


def test_scalar_ops():
    a = Value(5.0)
    b = Value(8.0)
    
    c = a + b
    assert np.allclose(c.data, 13.), PrintColor.FAIL + "Addition failed!" + PrintColor.ENDC
    c.backward()
    assert np.allclose(a.grad, 1.0), PrintColor.FAIL + "Addition backward failed for first operand!" + PrintColor.ENDC
    assert np.allclose(b.grad, 1.0), PrintColor.FAIL + "Addition backward failed for second operand!" + PrintColor.ENDC
    print(PrintColor.PASS + "Addition OK!" + PrintColor.ENDC)
    
    a = Value(5.0)
    b = Value(8.0)
    c = a * b
    assert np.allclose(c.data, 40.), PrintColor.FAIL + "Multiplication failed!" + PrintColor.ENDC
    c.backward()
    assert np.allclose(a.grad, 8.0), PrintColor.FAIL + "Multiplication backward for first operand failed!" + PrintColor.ENDC
    assert np.allclose(b.grad, 5.0), PrintColor.FAIL + "Multiplication backward for second operand failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "Multiplication OK!" + PrintColor.ENDC)
    
    a = Value(11.0)
    c = a - 8
    assert np.allclose(c.data, 3.0), PrintColor.FAIL + "Scalar subtraction failed!" + PrintColor.ENDC
    c.backward()
    assert np.allclose(a.grad, 1.0), PrintColor.FAIL + "Scalar subtraction backward failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "Scalar subtraction OK!" + PrintColor.ENDC)
    
    a = Value(18.0)
    b = Value(4.0)
    c = a / b
    assert np.allclose(c.data, 4.5), PrintColor.FAIL + "Division failed!" + PrintColor.ENDC
    c.backward()
    assert np.allclose(a.grad, 1 / 4.), PrintColor.FAIL + "Division backward for first operand failed!" + PrintColor.ENDC
    assert np.allclose(b.grad, (-18.0) / (4.0 ** 2)), PrintColor.FAIL + "Division backward for second operand failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "Division OK!" + PrintColor.ENDC)
    
    a = Value(3.0)
    b = 4
    c = a ** b
    assert np.allclose(c.data, 81.0), PrintColor.FAIL + "Power failed!" + PrintColor.ENDC
    c.backward()
    assert np.allclose(a.grad, 4 * (3.0 ** 3)), PrintColor.FAIL + "Power backward failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "Power OK!" + PrintColor.ENDC)
    
    print(PrintColor.PASS + "All scalar operators passed test!" + PrintColor.ENDC)
    print(DONE_TEST)
    
    
def test_broadcast_ops():
    # Matrix + scalar
    m = Value([[1, 2, 3], [4, 5, 6]])
    s = Value(10)
    c = m + s
    assert np.allclose(c.data, [[11, 12, 13], [14, 15, 16]]), PrintColor.FAIL + "Broadcasting (Matrix + Scalar) failed!" + PrintColor.ENDC
    c.backward()
    # Grad of matrix is 1s, grad of scalar is the sum of all incoming grads
    assert np.allclose(m.grad, np.ones((2, 3))), PrintColor.FAIL + "Broadcasting (Matrix + Scalar) backward failed for matrix!" + PrintColor.ENDC
    assert np.allclose(s.grad, 6), PrintColor.FAIL + "Broadcasting (Matrix + Scalar) backward failed for scalar!" + PrintColor.ENDC
    print(PrintColor.PASS + "Broadcasting (Matrix + Scalar) OK!" + PrintColor.ENDC)
    
    # Vector * Scalar
    v = Value([1, 2, 3])
    s = Value(5)
    c = v * s
    assert np.allclose(c.data, [5, 10, 15]), PrintColor.FAIL + "Broadcasting (Vector * Scalar) failed!" + PrintColor.ENDC
    c.backward()
    # grad of v is s.data, grad of s is sum of v.data
    assert np.allclose(v.grad, [5, 5, 5]), PrintColor.FAIL + "Broadcasting (Vector * Scalar) backward failed for vector!" + PrintColor.ENDC
    assert np.allclose(s.grad, 1 + 2 + 3), PrintColor.FAIL + "Broadcasting (Vector * Scalar) backward failed for scalar!" + PrintColor.ENDC
    print(PrintColor.PASS + "Broadcasting (Vector * Scalar) OK!" + PrintColor.ENDC)
    
    print(PrintColor.PASS + "All broadcasting operators passed test!" + PrintColor.ENDC)
    print(DONE_TEST)
    

def test_nonlin_ops():
    a = Value([-3, 0, 5])
    b = a.relu()
    assert np.allclose(b.data, [0, 0, 5]), PrintColor.FAIL + "ReLU failed!" + PrintColor.ENDC
    b.backward()
    assert np.allclose(a.grad, [0, 0, 1]), PrintColor.FAIL + "ReLU backward failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "ReLU OK!" + PrintColor.ENDC)

    a_data = np.array([-1, 0, 1])
    a = Value([-1, 0, 1])
    b = a.sigmoid()
    s = 1 / (1 + np.exp(-a_data))
    assert np.allclose(b.data, s), PrintColor.FAIL + "Sigmoid failed!" + PrintColor.ENDC
    b.backward()
    expected_grad = s * (1 - s)
    assert np.allclose(a.grad, expected_grad), PrintColor.FAIL + "Sigmoid backward failed!" + PrintColor.ENDC
    print(PrintColor.PASS + "Sigmoid OK!" + PrintColor.ENDC)
    
    print(PrintColor.PASS + "All non-linear operators passed test!" + PrintColor.ENDC)
    print(DONE_TEST)
    
    
def test_matmul():
    a_data = np.random.randn(2, 3)
    b_data = np.random.randn(3, 4)
    a = Value(a_data)
    b = Value(b_data)
    
    c = a @ b
    assert np.allclose(c.data, a_data @ b_data), PrintColor.FAIL + "Matmul failed!" + PrintColor.ENDC
    
    c.backward()
    assert np.allclose(a.grad, np.ones_like(c.data) @ b_data.T), PrintColor.FAIL + "Matmul backward failed for first operand!" + PrintColor.ENDC
    assert np.allclose(b.grad, a_data.T @ np.ones_like(c.data)), PrintColor.FAIL + "Matmul backward failed for second operand!" + PrintColor.ENDC
    
    print(PrintColor.PASS + "Malmul OK!" + PrintColor.ENDC)
    print(DONE_TEST)
    
    
def test_computational_graph():
    a = Value(2.0)
    b = Value(3.0)
    c = Value([10.0, 20.0])
    
    e = a * b
    d = e + c
    
    d.backward()
    
    # d/d=1, d/c=1
    assert np.allclose(d.grad, [1, 1]), PrintColor.FAIL + "Computational graph backward failed!" + PrintColor.ENDC
    assert np.allclose(c.grad, [1, 1]), PrintColor.FAIL + "Computational graph backward failed!" + PrintColor.ENDC
    # d/e = sum(d/d) (broadcasting) -> 1 + 1 = 2
    assert np.allclose(e.grad, 2), PrintColor.FAIL + "Computational graph backward failed!" + PrintColor.ENDC
    # # d/b = d/e * e/b = 2 * a = 2 * 2 = 4
    assert np.allclose(b.grad, 4.0), PrintColor.FAIL + "Computational graph backward failed!" + PrintColor.ENDC
    # d/a = d/e * e/a = 2 * b = 2 * 3 = 6
    assert np.allclose(a.grad, 6.0), PrintColor.FAIL + "Computational graph backward failed!" + PrintColor.ENDC
    
    print(PrintColor.PASS + "Computational graph OK!" + PrintColor.ENDC)
    print(DONE_TEST)
    
    
if __name__ == '__main__':
    test_scalar_ops()
    test_broadcast_ops()
    test_nonlin_ops()
    test_matmul()
    test_computational_graph()
