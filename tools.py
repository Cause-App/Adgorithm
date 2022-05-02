import numpy as np


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
    return np.concatenate([xb, yb], axis=-1)
