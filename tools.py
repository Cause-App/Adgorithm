import numpy as np
import tensorflow as tf


def factor_squarely(n):
    sqrt = np.sqrt(n)
    f, m = np.modf(n / (np.arange(int(sqrt))+1))
    f1 = int(m[f == 0][-1])
    f2 = n // f1
    assert f1 * f2 == n
    assert f1 >= f2
    return f1, f2


def style_corr_matrix(styler):
    return styler\
        .background_gradient(cmap='coolwarm')\
        .set_properties(**{"font-size": "0", "width": "10px", "height": "10px"})\
        .hide_columns()\
        .hide_index()


def ftr_cartesian_product(x, y):
    xl = x.shape[0]
    yl = y.shape[0]
    xb = np.swapaxes(np.broadcast_to(x, (yl, *x.shape)),
                     0, 1).reshape((xl*yl, *x.shape[1:]))
    yb = np.broadcast_to(y, (xl, *y.shape)).reshape((xl*yl, *y.shape[1:]))
    if len(x.shape) == len(y.shape):
        return np.concatenate([xb, yb], axis=-1)
    return (xb, yb)


def split_all(arrays, n):
    return (tuple(x[:n] for x in arrays), tuple(x[n:] for x in arrays))


def to_one_hot(x, min, max):
    return np.eye(max-min+1)[x-1]


def from_one_hot(x, min, max, mix=True):
    if mix:
        weights = tf.range(min, max+1, dtype="float32")
        return tf.reduce_sum(x * weights, axis=-1)
    else:
        return tf.math.argmax(x, axis=-1) + min
