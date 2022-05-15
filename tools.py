from IPython.display import display, update_display, HTML, Image
from tensorflow.keras.utils import plot_model
from keras.callbacks import Callback
import keras_tuner as kt
import tensorflow as tf
import numpy as np
import uuid
import os


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


def get_user_ad_ftr_product(user_ftrs, ad_ftrs):
    return (ftr_cartesian_product(user_ftrs, ad_ftrs[0]), ftr_cartesian_product(user_ftrs, ad_ftrs[1])[-1])


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


class HyperbandWithBatchSize(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Int(
            'batch_size', 32, 256, step=32)
        return super().run_trial(trial, *args, **kwargs)


class DisplayableCallback(Callback):
    def __init__(self, name, create_display_immediately=True, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.display_id = f"{self.name}-{str(uuid.uuid1())}"
        if create_display_immediately:
            self.create_display()

    def create_display(self):
        display(HTML(self.name), display_id=self.display_id)

    def update_display(self, obj):
        update_display(obj, display_id=self.display_id)


class ModelDisplayer(DisplayableCallback):
    def __init__(self, **kwargs):
        super().__init__("Model Topology", **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        model_plot_filename = "data/tmp.png"
        plot_model(self.model, to_file=model_plot_filename, show_shapes=True,
                   show_layer_activations=True, show_layer_names=True)
        self.update_display(Image(model_plot_filename))
        os.unlink(model_plot_filename)
        return super().on_epoch_end(epoch, logs=logs)


class Printer(DisplayableCallback):
    def __init__(self, **kwargs):
        super().__init__("Epoch Data", **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        header = f"Epoch {epoch}"
        lines = []
        if logs is not None:
            for k, v in logs.items():
                lines.append(f"{k}={v:.5f}")
        self.update_display(HTML(f"<h3>{header}</h3>"+"<br/>".join(lines)))
        return super().on_epoch_end(epoch, logs=logs)
