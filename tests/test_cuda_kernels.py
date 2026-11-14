"""Comprueba los kernels Numba contra NumPy. Se omite si no hay GPU CUDA o Numba."""
import unittest

import numpy as np

try:
    from asl_cpu.cuda_kernels import (adam_launch_config, adam_update, conv2d_3x3_bias_relu, conv2d_3x3_tiled,
                                      conv_tiled_config, cuda, grid_2d, matmul_bias_act, matmul_config,
                                      maxpool_2x2, softmax_config, softmax_rows)
    AVAILABLE = cuda.is_available()
except Exception:  # sin Numba, sin driver o en Windows CPU
    AVAILABLE = False


def conv_reference(x, w, b):
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
    return np.maximum(np.tensordot(windows, w, axes=([4, 5, 3], [0, 1, 2])) + b, 0).astype(np.float32)


@unittest.skipUnless(AVAILABLE, "requiere GPU CUDA y Numba")
class CudaKernelTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_both_convolutions_match_numpy_with_odd_sizes(self):
        # 20 × 18 no es múltiplo de 16 y 11 canales no es múltiplo de 8: prueba los bordes.
        x = self.rng.standard_normal((2, 20, 18, 11), dtype=np.float32)
        w = self.rng.standard_normal((3, 3, 11, 16), dtype=np.float32)
        b = self.rng.standard_normal(16, dtype=np.float32)
        expected = conv_reference(x, w, b)
        args = (cuda.to_device(x), cuda.to_device(w), cuda.to_device(b))
        for name, kernel, (grid, block) in (
                ("v1", conv2d_3x3_bias_relu, grid_2d(18, 20, 2 * 16)),
                ("v2", conv2d_3x3_tiled, conv_tiled_config(18, 20, 2, 16))):
            out = cuda.device_array(expected.shape, np.float32)
            kernel[grid, block](*args, out, 1)
            with self.subTest(kernel=name):
                np.testing.assert_allclose(out.copy_to_host(), expected, atol=1e-4, rtol=1e-4)

    def test_maxpool_matmul_and_softmax_match_numpy(self):
        x = self.rng.standard_normal((3, 8, 6, 5), dtype=np.float32)
        out = cuda.device_array((3, 4, 3, 5), np.float32)
        grid, block = grid_2d(3, 4, 15)
        maxpool_2x2[grid, block](cuda.to_device(x), out)
        np.testing.assert_array_equal(out.copy_to_host(), x.reshape(3, 4, 2, 3, 2, 5).max(axis=(2, 4)))

        a = self.rng.standard_normal((37, 70), dtype=np.float32)
        bmat = self.rng.standard_normal((70, 29), dtype=np.float32)
        bias = self.rng.standard_normal(29, dtype=np.float32)
        c = cuda.device_array((37, 29), np.float32)
        grid, block = matmul_config(37, 29)
        matmul_bias_act[grid, block](cuda.to_device(a), cuda.to_device(bmat), cuda.to_device(bias), c, 1)
        np.testing.assert_allclose(c.copy_to_host(), np.maximum(a @ bmat + bias, 0), atol=1e-4)

        z = np.ascontiguousarray(a[:, :29])
        probs = cuda.device_array_like(z)
        grid, block = softmax_config(37)
        softmax_rows[grid, block](cuda.to_device(z), probs)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        np.testing.assert_allclose(probs.copy_to_host(), e / e.sum(axis=1, keepdims=True), atol=1e-6)

    def test_adam_kernel_matches_keras_formula(self):
        n = 1000
        w, g = (self.rng.standard_normal(n, dtype=np.float32) for _ in range(2))
        m, v = (self.rng.random(n, dtype=np.float32) * 0.01 for _ in range(2))
        lr_t = np.float32(0.001 * np.sqrt(1 - 0.999 ** 3) / (1 - 0.9 ** 3))
        dm, dv, out = cuda.to_device(m), cuda.to_device(v), cuda.device_array(n, np.float32)
        blocks, threads = adam_launch_config(n)
        self.assertEqual((blocks, threads), (4, 256))
        adam_update[blocks, threads](cuda.to_device(w), cuda.to_device(g), dm, dv, out, lr_t,
                                     np.float32(0.9), np.float32(0.999), np.float32(1e-7))
        m_ref = m + (g - m) * np.float32(0.1)
        v_ref = v + (g * g - v) * np.float32(0.001)
        np.testing.assert_allclose(dm.copy_to_host(), m_ref, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(dv.copy_to_host(), v_ref, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(out.copy_to_host(), w - lr_t * m_ref / (np.sqrt(v_ref) + np.float32(1e-7)),
                                   rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
