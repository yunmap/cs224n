#-*- coding: utf-8 -*- 
import numpy as np
import tensorflow as tf
from utils.general_utils import test_all_close


def softmax(x):
    """
    Compute the softmax function in tensorflow.

    You might find the tensorflow functions tf.exp, tf.reduce_max,
    tf.reduce_sum, tf.expand_dims useful. (Many solutions are possible, so you may
    not need to use all of these functions). Recall also that many common
    tensorflow operations are sugared (e.g. x * y does a tensor multiplication
    if x and y are both tensors). Make sure to implement the numerical stability
    fixes as in the previous homework!

    Args:
        x:   tf.Tensor with shape (n_samples, n_features). Note feature vectors are
                  represented by row-vectors. (For simplicity, no need to handle 1-d
                  input as in the previous homework)
    Returns:
        out: tf.Tensor with shape (n_sample, n_features). You need to construct this
                  tensor in this problem.
    """

    ### YOUR CODE HERE
    #우선 x가 각 row가 한 sample을 나타내도록 되어 있음.
    #그래서 axis=1 로 sum과 max를 해야 각 sample에 대한 결과가 나온다.
    #keep_dims=true로 해둔 이유는 한 column의 형태로 각 sample에 대한 결과를 출력하기 위해.
    x_max = tf.reduce_max(x,1,keep_dims=True)
    x_sub = tf.subtract(x,x_max)
    #sub를 하면 max 값에서만 0이 되고, 나머지는 음수의 값을 가진다. 그래서 exp를 취하면 max(true class)는
    #1의 값을 갖고 나머지는 0에 가까운 값을 가지게 된다. (e^x 그래프 참고)
    #softmax의 과정
    x_exp = tf.exp(x_sub)
    sum_exp = tf.reduce_sum(x_exp,1,keep_dims=True)
    #sum_exp = tf.reduce_sum(x_exp)
    out = tf.div(x_exp,sum_exp)
    #out은 확률이라서
    ### END YOUR CODE

    return out


def cross_entropy_loss(y, yhat):
    """
    Compute the cross entropy loss in tensorflow.
    The loss should be summed over the current minibatch.

    y is a one-hot tensor of shape (n_samples, n_classes) and yhat is a tensor
    of shape (n_samples, n_classes). y should be of dtype tf.int32, and yhat should
    be of dtype tf.float32.

    The functions tf.to_float, tf.reduce_sum, and tf.log might prove useful. (Many
    solutions are possible, so you may not need to use all of these functions).

    Note: You are NOT allowed to use the tensorflow built-in cross-entropy
                functions.

    Args:
        y:    tf.Tensor with shape (n_samples, n_classes). One-hot encoded.
        yhat: tf.Tensor with shape (n_sample, n_classes). Each row encodes a
                    probability distribution and should sum to 1. 확률
    Returns:
        out:  tf.Tensor with shape (1,) (Scalar output). You need to construct this
                    tensor in the problem.
    """

    ### YOUR CODE HERE
    log_yhat = tf.log(yhat)
    #y_mul = tf.multiply(y, log_yhat) //y가 int32라고??? 머래는겨
    y_mul = tf.multiply(tf.to_float(y), log_yhat)
    #y_sum = tf.reduce_sum(y_mul,1,keep_dims=True)
    y_sum = tf.reduce_sum(y_mul)
    out = tf.multiply(-1.0, y_sum)
    ### END YOUR CODE

    return out


def test_softmax_basic():
    """
    Some simple tests of softmax to get you started.
    Warning: these are not exhaustive.
    """

    test1 = softmax(tf.constant(np.array([[1001, 1002], [3, 4]]), dtype=tf.float32))
    with tf.Session() as sess:
            test1 = sess.run(test1)
    test_all_close("Softmax test 1", test1, np.array([[0.26894142,  0.73105858],
                                                      [0.26894142,  0.73105858]]))

    test2 = softmax(tf.constant(np.array([[-1001, -1002]]), dtype=tf.float32))
    with tf.Session() as sess:
            test2 = sess.run(test2)
    test_all_close("Softmax test 2", test2, np.array([[0.73105858, 0.26894142]]))

    print "Basic (non-exhaustive) softmax tests pass\n"


def test_cross_entropy_loss_basic():
    """
    Some simple tests of cross_entropy_loss to get you started.
    Warning: these are not exhaustive.
    """
    y = np.array([[0, 1], [1, 0], [1, 0]])
    yhat = np.array([[.5, .5], [.5, .5], [.5, .5]])

    test1 = cross_entropy_loss(
            tf.constant(y, dtype=tf.int32),
            tf.constant(yhat, dtype=tf.float32))
    with tf.Session() as sess:
        test1 = sess.run(test1)
    expected = -3 * np.log(.5)
    test_all_close("Cross-entropy test 1", test1, expected)

    print "Basic (non-exhaustive) cross-entropy tests pass"

if __name__ == "__main__":
    test_softmax_basic()
    test_cross_entropy_loss_basic()
