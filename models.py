from keras.layers import Input, MultiHeadAttention, Dense, Concatenate, GlobalAveragePooling1D, Dropout, Lambda, Layer, LayerNormalization
from keras.models import Model, Sequential
from keras.regularizers import L1L2
import tensorflow as tf


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim), ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


def rating_mae(y_true, y_pred):
    r_true = tools.from_one_hot(y_true, 1, 5, mix=False)
    r_pred = tools.from_one_hot(y_pred, 1, 5, mix=False)
    return tf.keras.metrics.mean_absolute_error(r_true, r_pred)


def mixed_rating_mae(y_true, y_pred):
    r_true = tools.from_one_hot(y_true, 1, 5, mix=True)
    r_pred = tools.from_one_hot(y_pred, 1, 5, mix=True)
    return tf.keras.metrics.mean_absolute_error(r_true, r_pred)


def create_uai_model(ad_ftrs, user_ftrs, ratings, hp=None):
    inputs_1 = Input(
        (user_ftrs.shape[1]+ad_ftrs[0].shape[1]), name="flat_input")
    inputs_2 = Input(ad_ftrs[1].shape[1:], name="text_input")

    input_dropout_rate = 0.5 if hp is None else hp.Choice(
        "input_dropout", values=[0.0, 0.1, 0.3, 0.5])
    subnet_1 = inputs_1 if input_dropout_rate == 0 else Dropout(
        input_dropout_rate, name="flat_input_dropout")(inputs_1)
    subnet_2 = inputs_2 if input_dropout_rate == 0 else Dropout(
        input_dropout_rate, name="text_input_dropout")(inputs_2)

    l1 = 1e-2 if hp is None else hp.Choice("l1", values=[1e-2, 1e-3, 1e-4])
    l2 = 1e-2 if hp is None else hp.Choice("l2", values=[1e-2, 1e-3, 1e-4])
    reg = L1L2(l1=l1, l2=l2)

    num_heads = 2 if hp is None else hp.Int(
        "num_attn_heads", min_value=1, max_value=8, step=1)
    ff_dim = 32 if hp is None else hp.Int(
        "ff_dim", min_value=32, max_value=128, step=32)
    dropout_rate = 0.5 if hp is None else hp.Choice(
        "transformer_dropout_1", values=[0.0, 0.1, 0.3, 0.5])
    subnet_2 = TransformerBlock(embed_dim=ad_ftrs[1].shape[-1], num_heads=num_heads,
                                ff_dim=ff_dim, rate=dropout_rate, name="transformer_block")(subnet_2)
    subnet_2 = GlobalAveragePooling1D(name="pooling")(subnet_2)
    dropout_rate = 0.5 if hp is None else hp.Choice(
        "transformer_dropout_2", values=[0.0, 0.1, 0.3, 0.5])
    if dropout_rate != 0:
        subnet_2 = Dropout(dropout_rate)(subnet_2)

    output = Concatenate(name="combined_subnets")([subnet_1, subnet_2])

    num_hidden_layers = 2 if hp is None else hp.Int("num_hidden_layers", 1, 3)
    for i in range(num_hidden_layers):
        units = 32 if hp is None else hp.Int(
            f"units_{i}", min_value=32, max_value=128, step=32)
        output = Dense(units, activation="relu",
                       kernel_regularizer=reg, name=f"hidden_{i}")(output)
        dropout_rate = 0.5 if hp is None else hp.Choice(
            f"dropout_{i}", values=[0.0, 0.1, 0.3, 0.5])
        if dropout_rate != 0:
            output = Dropout(dropout_rate, name=f"hidden_dropout_{i}")(output)

    output = Dense(5, kernel_regularizer=reg, name="output",
                   activation="softmax")(output)

    model = Model(inputs=[inputs_1, inputs_2], outputs=[output])
    learning_rate = 1e-2 if hp is None else hp.Choice(
        "learning_rate", values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", rating_mae, mixed_rating_mae]
    )

    return model


def mean_binary_crossentropy(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    losses = tf.map_fn(
        fn=lambda x: bce(x[0], x[1]),
        elems=[tf.transpose(y_true), tf.transpose(y_pred)],
        fn_output_signature=tf.float32
    )
    return tf.reduce_mean(losses)


def link_up_rl_model(hp, inputs_1, inputs_2, ad_ftrs, idx=None):
    idx = "" if idx is None else f"_{idx}"
    input_dropout_rate = 0.5 if hp is None else hp.Choice(
        "input_dropout", values=[0.0, 0.1, 0.3, 0.5])
    subnet_1 = inputs_1 if input_dropout_rate == 0 else Dropout(
        input_dropout_rate, name=f"flat_input_dropout{idx}")(inputs_1)
    subnet_2 = inputs_2 if input_dropout_rate == 0 else Dropout(
        input_dropout_rate, name=f"text_input_dropout{idx}")(inputs_2)

    l1 = 1e-2 if hp is None else hp.Choice("l1", values=[1e-2, 1e-3, 1e-4])
    l2 = 1e-2 if hp is None else hp.Choice("l2", values=[1e-2, 1e-3, 1e-4])
    reg = L1L2(l1=l1, l2=l2)

    num_heads = 2 if hp is None else hp.Int(
        "num_attn_heads", min_value=1, max_value=8, step=1)
    ff_dim = 32 if hp is None else hp.Int(
        "ff_dim", min_value=32, max_value=128, step=32)
    dropout_rate = 0.5 if hp is None else hp.Choice(
        "transformer_dropout_1", values=[0.0, 0.1, 0.3, 0.5])
    subnet_2 = TransformerBlock(embed_dim=ad_ftrs[1].shape[-1], num_heads=num_heads,
                                ff_dim=ff_dim, rate=dropout_rate, name=f"transformer_block{idx}")(subnet_2)
    subnet_2 = GlobalAveragePooling1D(name=f"pooling{idx}")(subnet_2)
    dropout_rate = 0.5 if hp is None else hp.Choice(
        "transformer_dropout_2", values=[0.0, 0.1, 0.3, 0.5])
    if dropout_rate != 0:
        subnet_2 = Dropout(dropout_rate)(subnet_2)

    output = Concatenate(name=f"combined_subnets{idx}")([subnet_1, subnet_2])

    num_hidden_layers = 2 if hp is None else hp.Int("num_hidden_layers", 1, 3)
    for i in range(num_hidden_layers):
        units = 32 if hp is None else hp.Int(
            f"units_{i}", min_value=32, max_value=128, step=32)
        output = Dense(units, activation="relu",
                       kernel_regularizer=reg, name=f"hidden_{i}{idx}")(output)
        dropout_rate = 0.5 if hp is None else hp.Choice(
            f"dropout_{i}", values=[0.0, 0.1, 0.3, 0.5])
        if dropout_rate != 0:
            output = Dropout(
                dropout_rate, name=f"hidden_dropout_{i}{idx}")(output)

    output = Dense(1, activation="sigmoid",
                   kernel_regularizer=reg, name=f"output{idx}")(output)
    return output


def create_rl_model(ad_ftrs, hp=None, metrics=[], hypertuning=False, n_users=1):
    assert n_users == 1 or hypertuning

    learning_rate = 1e-2 if hp is None else hp.Choice(
        "learning_rate", values=[1e-2, 1e-3, 1e-4])

    if hypertuning:
        big_inputs_1 = Input(
            (n_users, ad_ftrs[0].shape[1]+1,), name="flat_input_stacked")
        big_inputs_2 = Input(
            (n_users, *ad_ftrs[1].shape[1:]), name="text_input_stacked")

        outputs = []

        for i in range(n_users):
            inputs_1 = Lambda(lambda x: x[:, i],
                              name=f"flat_input_{i}")(big_inputs_1)
            inputs_2 = Lambda(lambda x: x[:, i],
                              name=f"text_input_{i}")(big_inputs_2)

            output = link_up_rl_model(hp, inputs_1, inputs_2, ad_ftrs, idx=i)
            outputs.append(output)

        output = Concatenate(name="output")(outputs)

        model = Model(inputs=[big_inputs_1, big_inputs_2],
                      outputs=[output, output])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=[mean_binary_crossentropy, "mse"],
            loss_weights=[1, 0],
            metrics=metrics
        )
        return model

    else:
        inputs_1 = Input((ad_ftrs[0].shape[1]+1,), name="flat_input")
        inputs_2 = Input(ad_ftrs[1].shape[1:], name="text_input")

        output = link_up_rl_model(hp, inputs_1, inputs_2, ad_ftrs)

        model = Model(inputs=[inputs_1, inputs_2], outputs=[output])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=metrics
        )
        return model
