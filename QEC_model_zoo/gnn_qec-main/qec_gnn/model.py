"""TensorFlow GCN model for graph-level logical-flip classification."""

from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="qec_gnn")
class GCNLayer(tf.keras.layers.Layer):
    """Dense GCN layer: ``ReLU(A_norm @ X @ W + b)``."""

    def __init__(self, units: int, activation: str | None = "relu", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shapes):
        feature_dim = int(input_shapes[0][-1])
        self.w = self.add_weight(
            name="w",
            shape=(feature_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        x, a = inputs
        h = tf.matmul(x, self.w)
        h = tf.matmul(a, h)
        h = h + self.b
        return self.activation(h) if self.activation is not None else h

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "activation": self.activation_name})
        return config


@tf.keras.utils.register_keras_serializable(package="qec_gnn")
class MaskedMeanPool(tf.keras.layers.Layer):
    """Mean-pool node embeddings while ignoring padded nodes."""

    def call(self, inputs):
        h, mask = inputs
        mask = tf.cast(mask, h.dtype)
        mask = tf.expand_dims(mask, axis=-1)
        h = h * mask
        return tf.reduce_sum(h, axis=1) / (tf.reduce_sum(mask, axis=1) + tf.keras.backend.epsilon())


@tf.keras.utils.register_keras_serializable(package="qec_gnn")
class MaskedMaxPool(tf.keras.layers.Layer):
    """Max-pool node embeddings while ignoring padded nodes."""

    def call(self, inputs):
        h, mask = inputs
        mask = tf.cast(mask, h.dtype)
        mask = tf.expand_dims(mask, axis=-1)
        very_negative = tf.cast(-1e9, h.dtype)
        masked = tf.where(mask > 0, h, very_negative)
        return tf.reduce_max(masked, axis=1)


@tf.keras.utils.register_keras_serializable(package="qec_gnn")
class MaskedFlatten(tf.keras.layers.Layer):
    """Flatten masked node embeddings to make tiny-overfit memorization possible."""

    def call(self, inputs):
        h, mask = inputs
        mask = tf.cast(mask, h.dtype)
        h = h * tf.expand_dims(mask, axis=-1)
        return tf.reshape(h, (tf.shape(h)[0], -1))

    def compute_output_shape(self, input_shapes):
        h_shape = input_shapes[0]
        return (h_shape[0], h_shape[1] * h_shape[2])


def build_gnn_model(
    *,
    max_nodes: int,
    feature_dim: int,
    hidden_dim: int = 64,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build and compile the minimal dense GCN decoder."""
    x_in = tf.keras.Input(shape=(max_nodes, feature_dim), name="X")
    a_in = tf.keras.Input(shape=(max_nodes, max_nodes), name="A")
    mask_in = tf.keras.Input(shape=(max_nodes,), name="mask")

    h = GCNLayer(hidden_dim, name="gcn1")([x_in, a_in])
    h = GCNLayer(hidden_dim, name="gcn2")([h, a_in])
    h = GCNLayer(hidden_dim, name="gcn3")([h, a_in])
    mean_pool = MaskedMeanPool(name="masked_mean_pool")([h, mask_in])
    max_pool = MaskedMaxPool(name="masked_max_pool")([h, mask_in])
    flat_pool = MaskedFlatten(name="masked_flatten")([h, mask_in])
    raw_flat = MaskedFlatten(name="raw_detector_summary")([x_in, mask_in])
    raw_summary = tf.keras.layers.Dense(hidden_dim, activation="relu", name="raw_summary_dense")(raw_flat)
    graph_embedding = tf.keras.layers.Concatenate(name="graph_summary")(
        [mean_pool, max_pool, flat_pool, raw_summary]
    )
    graph_embedding = tf.keras.layers.Dropout(dropout, name="dropout")(graph_embedding)
    graph_embedding = tf.keras.layers.Dense(hidden_dim, activation="relu", name="dense1")(graph_embedding)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="logical_flip")(graph_embedding)

    model = tf.keras.Model(inputs=[x_in, a_in, mask_in], outputs=output, name="gnn_decoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def residual_gcn_block(
    h: tf.Tensor,
    a: tf.Tensor,
    *,
    hidden_dim: int,
    dropout: float,
    name: str,
) -> tf.Tensor:
    """Residual GCN block with normalization and dropout."""
    shortcut = h
    h = GCNLayer(hidden_dim, name=f"{name}_gcn")([h, a])
    h = tf.keras.layers.LayerNormalization(name=f"{name}_norm")(h)
    h = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")(h)

    if int(shortcut.shape[-1]) != hidden_dim:
        shortcut = tf.keras.layers.Dense(hidden_dim, name=f"{name}_projection")(shortcut)

    h = tf.keras.layers.Add(name=f"{name}_residual")([shortcut, h])
    return tf.keras.layers.Activation("relu", name=f"{name}_activation")(h)


def build_residual_gnn_model(
    *,
    max_nodes: int,
    feature_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    num_blocks: int = 4,
) -> tf.keras.Model:
    """Build and compile the stronger fixed-graph residual GCN decoder."""
    x_in = tf.keras.Input(shape=(max_nodes, feature_dim), name="X")
    a_in = tf.keras.Input(shape=(max_nodes, max_nodes), name="A")
    mask_in = tf.keras.Input(shape=(max_nodes,), name="mask")

    h = tf.keras.layers.Dense(hidden_dim, activation="relu", name="input_projection")(x_in)
    for block_id in range(num_blocks):
        h = residual_gcn_block(
            h,
            a_in,
            hidden_dim=hidden_dim,
            dropout=dropout,
            name=f"res_gcn_{block_id + 1}",
        )

    mean_pool = MaskedMeanPool(name="masked_mean_pool")([h, mask_in])
    max_pool = MaskedMaxPool(name="masked_max_pool")([h, mask_in])
    flat_pool = MaskedFlatten(name="masked_flatten")([h, mask_in])
    graph_embedding = tf.keras.layers.Concatenate(name="graph_summary")([mean_pool, max_pool, flat_pool])
    graph_embedding = tf.keras.layers.Dense(hidden_dim, activation="relu", name="dense1")(graph_embedding)
    graph_embedding = tf.keras.layers.Dropout(dropout, name="head_dropout")(graph_embedding)
    graph_embedding = tf.keras.layers.Dense(hidden_dim // 2, activation="relu", name="dense2")(graph_embedding)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="logical_flip")(graph_embedding)

    model = tf.keras.Model(inputs=[x_in, a_in, mask_in], outputs=output, name="residual_gnn_decoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_decoder_model(
    *,
    model_type: str,
    max_nodes: int,
    feature_dim: int,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
) -> tf.keras.Model:
    """Build a decoder model by name."""
    if model_type == "gcn":
        return build_gnn_model(
            max_nodes=max_nodes,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
        )
    if model_type == "residual_gcn":
        return build_residual_gnn_model(
            max_nodes=max_nodes,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def load_gnn_model(path: str):
    """Load a saved model with registered custom GCN layers available."""
    return tf.keras.models.load_model(path)
