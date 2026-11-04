"""Arquitectura de la CNN base reutilizable en CPU y GPU."""


def create_cnn_base(tf, config: dict, number_of_classes: int = 29):
    """Reproduce crear_cnn_base de la celda 59 del notebook original."""
    layers = tf.keras.layers
    return tf.keras.Sequential([
        layers.Input((config["image_size"], config["image_size"], 3)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(number_of_classes, activation="softmax"),
    ], name="cnn_base")
