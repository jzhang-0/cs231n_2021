import numpy as np
from cs231n.classifiers.fc_net import *


def rel_error(x, y):
    """Returns relative error."""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x)  # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension

    return grad


# np.random.seed(231)
# N, D, H1, H2, C = 2, 15, 20, 30, 10
# X = np.random.randn(N, D)
# y = np.random.randint(C, size=(N,))

# for dropout_keep_ratio in [1, 0.75, 0.5]:
#     print('Running check with dropout = ', dropout_keep_ratio)
#     model = FullyConnectedNet(
#         [H1, H2],
#         input_dim=D,
#         num_classes=C,
#         weight_scale=5e-2,
#         dtype=np.float64,
#         dropout_keep_ratio=dropout_keep_ratio,
#         seed=123
#     )

#     loss, grads = model.loss(X, y)
#     print('Initial loss: ', loss)

#     # Relative errors should be around e-6 or less.
#     # Note that it's fine if for dropout_keep_ratio=1 you have W2 error be on the order of e-5.
#     for name in sorted(grads):
#         f = lambda _: model.loss(X, y)[0]
#         grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
#         print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
#     print()

# x_shape = (2, 3, 4, 4)
# w_shape = (3, 3, 4, 4)
# x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
# w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
# b = np.linspace(-0.1, 0.2, num=3)

# conv_param = {'stride': 2, 'pad': 1}
# out, _ = conv_forward_naive(x, w, b, conv_param)
# correct_out = np.array([[[[-0.08759809, -0.10987781],
#                            [-0.18387192, -0.2109216 ]],
#                           [[ 0.21027089,  0.21661097],
#                            [ 0.22847626,  0.23004637]],
#                           [[ 0.50813986,  0.54309974],
#                            [ 0.64082444,  0.67101435]]],
#                          [[[-0.98053589, -1.03143541],
#                            [-1.19128892, -1.24695841]],
#                           [[ 0.69108355,  0.66880383],
#                            [ 0.59480972,  0.56776003]],
#                           [[ 2.36270298,  2.36904306],
#                            [ 2.38090835,  2.38247847]]]])

# # Compare your output to ours; difference should be around e-8
# print('Testing conv_forward_naive')
# print('difference: ', rel_error(out, correct_out))

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad

# np.random.seed(231)
# x = np.random.randn(4, 3, 5, 5)
# w = np.random.randn(2, 3, 3, 3)
# b = np.random.randn(2,)
# dout = np.random.randn(4, 2, 5, 5)
# conv_param = {'stride': 1, 'pad': 1}

# dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
# dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
# db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)

# out, cache = conv_forward_naive(x, w, b, conv_param)
# dx, dw, db = conv_backward_naive(dout, cache)

# # Your errors should be around e-8 or less.
# print('Testing conv_backward_naive function')
# print('dx error: ', rel_error(dx, dx_num))
# print('dw error: ', rel_error(dw, dw_num))
# print('db error: ', rel_error(db, db_num))

# np.random.seed(231)
# x = np.random.randn(3, 2, 8, 8)
# dout = np.random.randn(3, 2, 4, 4)
# pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

# dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward_naive(x, pool_param)[0], x, dout)

# out, cache = max_pool_forward_naive(x, pool_param)
# dx = max_pool_backward_naive(dout, cache)

# # Your error should be on the order of e-12
# print('Testing max_pool_backward_naive function:')
# print('dx error: ', rel_error(dx, dx_num))

from cs231n.classifiers.cnn import *
model = ThreeLayerConvNet()

N = 50
X = np.random.randn(N, 3, 32, 32)
y = np.random.randint(10, size=N)

loss, grads = model.loss(X, y)
print('Initial loss (no regularization): ', loss)

model.reg = 0.5
loss, grads = model.loss(X, y)
print('Initial loss (with regularization): ', loss)