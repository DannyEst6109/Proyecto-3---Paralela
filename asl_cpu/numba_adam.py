"""Entrenamiento GPU con TensorFlow para forward/gradientes y Adam como kernel Numba propio.

Flujo de un paso (todo en la GPU, sin pasar por la CPU):

1. TensorFlow (tf.function, cuDNN/cuBLAS) calcula probabilidades, pérdida y gradientes.
2. Los gradientes y pesos se exponen a CuPy/Numba con DLPack (sin copia).
3. `adam_update` (cuda_kernels.py) actualiza cada tensor de pesos: un hilo por parámetro,
   bloques de 256 hilos, grid de ceil(n / 256) bloques. Los momentos m y v viven en la GPU.
4. El resultado vuelve a TensorFlow con DLPack y se asigna a las variables del modelo.

TensorFlow y Numba usan colas (streams) CUDA distintas, por eso se sincroniza antes de leer
los gradientes y después del kernel. Ese costo es parte de lo que se mide.
"""
from __future__ import annotations

import math
from time import perf_counter

import numpy as np


class NumbaAdam:
    """Adam de Keras (β1=0.9, β2=0.999, ε=1e-7) implementado con el kernel adam_update."""

    def __init__(self, tf, variables, learning_rate: float, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        import cupy as cp

        from .cuda_kernels import adam_launch_config, adam_update, cuda

        self.tf, self.cp, self.cuda = tf, cp, cuda
        self.kernel = adam_update
        self.variables = list(variables)
        self.learning_rate, self.beta_1, self.beta_2, self.epsilon = learning_rate, beta_1, beta_2, epsilon
        sizes = [int(np.prod(v.shape)) for v in self.variables]
        self.m = [cp.zeros(n, dtype=cp.float32) for n in sizes]
        self.v = [cp.zeros(n, dtype=cp.float32) for n in sizes]
        self.out = [cp.empty(n, dtype=cp.float32) for n in sizes]
        self.configs = [adam_launch_config(n) for n in sizes]
        self.iterations = 0
        self.kernel_seconds = 0.0
        self.interop_seconds = 0.0

    def launch_table(self) -> list[dict]:
        return [{"variable": v.path if hasattr(v, "path") else v.name, "parameters": int(np.prod(v.shape)),
                 "grid": blocks, "block": threads, "total_threads": blocks * threads}
                for v, (blocks, threads) in zip(self.variables, self.configs)]

    def state_device(self) -> str:
        return f"cupy/numba device {self.cp.cuda.Device().id} ({self.cuda.get_current_device().name.decode()})"

    def apply(self, flat_grads, flat_weights) -> None:
        tf, cp = self.tf, self.cp
        self.iterations += 1
        t = self.iterations
        lr_t = np.float32(self.learning_rate * math.sqrt(1 - self.beta_2 ** t) / (1 - self.beta_1 ** t))
        b1, b2, eps = np.float32(self.beta_1), np.float32(self.beta_2), np.float32(self.epsilon)
        start = perf_counter()
        tf.test.experimental.sync_devices()                 # gradientes listos en la cola de TF
        grads = [cp.from_dlpack(tf.experimental.dlpack.to_dlpack(g)) for g in flat_grads]
        weights = [cp.from_dlpack(tf.experimental.dlpack.to_dlpack(w)) for w in flat_weights]
        middle = perf_counter()
        for w, g, m, v, out, (blocks, threads) in zip(weights, grads, self.m, self.v, self.out, self.configs):
            self.kernel[blocks, threads](w, g, m, v, out, lr_t, b1, b2, eps)
        self.cuda.synchronize()                             # kernel terminado antes de que TF lea
        after_kernel = perf_counter()
        # __dlpack__() sin max_version entrega la cápsula "dltensor" clásica que acepta TF.
        for variable, out in zip(self.variables, self.out):
            variable.assign(tf.reshape(tf.experimental.dlpack.from_dlpack(out.__dlpack__()), variable.shape))
        self.kernel_seconds += after_kernel - middle
        self.interop_seconds += (middle - start) + (perf_counter() - after_kernel)


def fit_with_numba_adam(tf, model, datasets, epochs: int, learning_rate: float, callbacks, verbose=True):
    """Equivalente a model.fit(train, validation_data=val) con la actualización hecha por Numba."""
    optimizer = NumbaAdam(tf, model.trainable_variables, learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    variables = model.trainable_variables

    @tf.function(jit_compile=False, reduce_retracing=True)
    def gradients(x, y):
        with tf.GradientTape() as tape:
            probabilities = model(x, training=True)
            loss = loss_fn(y, probabilities)
        grads = tape.gradient(loss, variables)
        correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(probabilities, axis=1, output_type=tf.int32),
                                                 tf.cast(y, tf.int32)), tf.float32))
        return (loss, correct, [tf.reshape(g, [-1]) for g in grads],
                [tf.reshape(v, [-1]) for v in variables])

    callback_list = tf.keras.callbacks.CallbackList(callbacks, model=model)
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    callback_list.on_train_begin()
    for epoch in range(epochs):
        callback_list.on_epoch_begin(epoch)
        start = perf_counter()
        loss_sum = correct_sum = 0.0
        seen = 0
        steps = 0
        for x, y in datasets["train"]:
            loss, correct, flat_grads, flat_weights = gradients(x, y)
            optimizer.apply(flat_grads, flat_weights)
            batch = int(y.shape[0])
            loss_sum += float(loss) * batch     # pérdida media ponderada por lote, como Keras
            correct_sum += float(correct)
            seen += batch
            steps += 1
        validation = model.evaluate(datasets["val"], return_dict=True, verbose=0)
        logs = {"accuracy": correct_sum / seen, "loss": loss_sum / seen,
                "val_accuracy": float(validation["accuracy"]), "val_loss": float(validation["loss"])}
        for key, value in logs.items():
            history[key].append(value)
        if verbose:
            elapsed = perf_counter() - start
            print(f"Epoch {epoch + 1}/{epochs}\n{steps}/{steps} - {elapsed:.0f}s - "
                  f"{1000 * elapsed / steps:.0f}ms/step - accuracy: {logs['accuracy']:.4f} - "
                  f"loss: {logs['loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f} - "
                  f"val_loss: {logs['val_loss']:.4f}", flush=True)
        callback_list.on_epoch_end(epoch, logs)
    callback_list.on_train_end()
    return history, optimizer
