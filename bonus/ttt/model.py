import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_policy(hidden: int = 64) -> keras.Model:
    inp = keras.Input(shape=(9,), dtype="float32")
    x = layers.Dense(hidden, activation="tanh")(inp)
    x = layers.Dense(hidden, activation="tanh")(x)
    logits = layers.Dense(9, activation=None)(x)  # unnormalized
    return keras.Model(inputs=inp, outputs=logits, name="ttt_policy")
