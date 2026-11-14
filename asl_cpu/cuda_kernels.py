"""Kernels CUDA propios escritos en CUDA Python (Numba) para la CNN base.

Cada kernel documenta qué calcula un hilo y cómo se forman los bloques y el grid.
Son la parte del diseño paralelo implementada por el equipo; el entrenamiento completo
usa además TensorFlow (cuDNN/cuBLAS) para el forward y los gradientes.

| Kernel                 | Un hilo calcula                      | Bloque        | Grid                                   |
|------------------------|--------------------------------------|---------------|----------------------------------------|
| adam_update            | un parámetro del modelo              | 256           | ceil(n / 256)                          |
| conv2d_3x3_bias_relu   | un valor de salida (n, y, x, filtro) | 16 × 16       | (ceil(W/16), ceil(H/16), N × filtros)  |
| conv2d_3x3_tiled (v2)  | 8 filtros de un pixel; tesela de     | 16 × 16       | (ceil(W/16), ceil(H/16), N × filtros/8)|
|                        | entrada 18×18×8 en memoria compartida                                                  |
| maxpool_2x2            | un valor de salida (n, y, x, canal)  | 16 × 16       | (ceil(W/2/16), ceil(H/2/16), N × C)    |
| matmul_bias_act        | un elemento de C = A·B + sesgo       | 16 × 16       | (ceil(cols/16), ceil(filas/16))        |
|                        | (teselas 16 × 16 en memoria compartida)                                                |
| softmax_rows           | una fila (una imagen, 29 clases)     | 64            | ceil(filas / 64)                       |
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path


def setup_numba_cuda() -> dict:
    """Hace que Numba use el compilador NVVM instalado por `tensorflow[and-cuda]`.

    El paquete pip nvidia-cuda-nvcc trae `nvvm/lib64/libnvvm.so` sin enlace versionado y
    Numba solo busca `libnvvm.so.N`. Se fija CUDA_HOME (para libdevice) y la ruta de NVVM.
    """
    import importlib.util

    spec = importlib.util.find_spec("nvidia")
    if spec is None or sys.platform != "linux":
        return {}
    for base in spec.submodule_search_locations:
        nvcc = Path(base) / "cuda_nvcc"
        nvvm = nvcc / "nvvm" / "lib64" / "libnvvm.so"
        if nvvm.is_file():
            os.environ.setdefault("CUDA_HOME", str(nvcc))
            from numba.cuda import cuda_paths

            paths = cuda_paths.get_cuda_paths()
            if paths["nvvm"].info is None:
                paths["nvvm"] = cuda_paths._env_path_tuple("pip nvidia-cuda-nvcc", str(nvvm))
            runtime = Path(base) / "cuda_runtime" / "lib"
            if runtime.is_dir():
                paths["cudalib_dir"] = cuda_paths._env_path_tuple("pip nvidia-cuda-runtime", str(runtime))
            # NVVM 12.9 emite PTX 8.8 y un driver de la serie 12.4 solo compila hasta PTX 8.4.
            # pynvjitlink compila el PTX a cubin dentro del proceso con nvJitLink, así el driver
            # recibe código binario (compatibilidad de versión menor de CUDA 12).
            linker = "driver"
            try:
                from pynvjitlink.patch import patch_numba_linker

                patch_numba_linker()
                linker = "pynvjitlink"
            except ImportError:
                pass
            return {"nvvm": str(paths["nvvm"].info), "cuda_home": os.environ["CUDA_HOME"],
                    "cudart_dir": str(paths["cudalib_dir"].info), "linker": linker}
    return {}


NUMBA_SETUP = setup_numba_cuda()

from numba import cuda, float32, int32  # noqa: E402  (después de configurar NVVM)

TILE = 16              # lado de la tesela de matmul y de los bloques 2D
ADAM_THREADS = 256     # hilos por bloque de los kernels 1D
SOFTMAX_THREADS = 64


# ---------------------------------------------------------------- Adam (entrenamiento)
@cuda.jit("void(float32[:], float32[:], float32[:], float32[:], float32[:], float32, float32, float32, float32)")
def adam_update(w, g, m, v, w_out, lr_t, beta_1, beta_2, epsilon):
    """Un hilo actualiza un parámetro. Misma fórmula que keras.optimizers.Adam:

    m ← m + (g − m)(1 − β1);  v ← v + (g² − v)(1 − β2);  w ← w − lr_t · m / (√v + ε)
    con lr_t = lr · √(1 − β2^t) / (1 − β1^t) calculado una vez por paso en el host.
    """
    i = cuda.grid(1)                      # blockIdx.x * blockDim.x + threadIdx.x
    if i < w.size:                        # el último bloque puede tener hilos sobrantes
        gi = g[i]
        mi = m[i] + (gi - m[i]) * (float32(1.0) - beta_1)
        vi = v[i] + (gi * gi - v[i]) * (float32(1.0) - beta_2)
        m[i] = mi
        v[i] = vi
        w_out[i] = w[i] - lr_t * mi / (math.sqrt(vi) + epsilon)


def adam_launch_config(n: int) -> tuple[int, int]:
    return (n + ADAM_THREADS - 1) // ADAM_THREADS, ADAM_THREADS


# ------------------------------------------------------- forward de la CNN (inferencia)
@cuda.jit("void(float32[:, :, :, :], float32[:, :, :, :], float32[:], float32[:, :, :, :], int32)")
def conv2d_3x3_bias_relu(x, w, b, out, relu):
    """Convolución 3×3, stride 1, padding same, NHWC, con sesgo y ReLU fusionados.

    threadIdx.x/y recorren columnas y filas de la imagen de salida; blockIdx.z
    codifica (imagen, filtro). Cada hilo suma 3 × 3 × C_in productos.
    """
    ox, oy = cuda.grid(2)
    z = cuda.blockIdx.z
    c_out = out.shape[3]
    n = z // c_out
    co = z % c_out
    height, width, c_in = x.shape[1], x.shape[2], x.shape[3]
    if ox >= width or oy >= height:
        return
    acc = b[co]
    for ky in range(3):
        iy = oy + ky - 1
        if iy < 0 or iy >= height:
            continue
        for kx in range(3):
            ix = ox + kx - 1
            if ix < 0 or ix >= width:
                continue
            for ci in range(c_in):
                acc += x[n, iy, ix, ci] * w[ky, kx, ci, co]
    if relu and acc < 0:
        acc = float32(0.0)
    out[n, oy, ox, co] = acc


CI_CHUNK = 8     # canales de entrada por tesela en memoria compartida
CO_PER_THREAD = 8  # filtros de salida que acumula cada hilo en registros
HALO = TILE + 2  # 16 + borde de 1 pixel a cada lado para el filtro 3×3


@cuda.jit("void(float32[:, :, :, :], float32[:, :, :, :], float32[:], float32[:, :, :, :], int32)")
def conv2d_3x3_tiled(x, w, b, out, relu):
    """Versión 2 de la convolución: teselas en memoria compartida y 8 filtros por hilo.

    Bloque 16 × 16 = tesela de salida de 16 × 16 pixeles de una imagen. blockIdx.z codifica
    (imagen, grupo de 8 filtros). Por cada grupo de 8 canales de entrada, los 256 hilos copian
    juntos a memoria compartida la tesela de entrada de 18 × 18 × 8 (con borde) y los pesos
    3 × 3 × 8 × 8; luego cada hilo acumula sus 8 salidas leyendo solo memoria compartida.
    Cada valor de entrada leído de memoria global se reutiliza 9 × 8 = 72 veces.
    """
    tile = cuda.shared.array(shape=(HALO, HALO, CI_CHUNK), dtype=float32)
    sw = cuda.shared.array(shape=(3, 3, CI_CHUNK, CO_PER_THREAD), dtype=float32)
    acc = cuda.local.array(CO_PER_THREAD, dtype=float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    tid = ty * TILE + tx
    height, width, c_in = x.shape[1], x.shape[2], x.shape[3]
    c_out = out.shape[3]
    groups = (c_out + CO_PER_THREAD - 1) // CO_PER_THREAD
    n = cuda.blockIdx.z // groups
    co0 = (cuda.blockIdx.z % groups) * CO_PER_THREAD
    ox = cuda.blockIdx.x * TILE + tx
    oy = cuda.blockIdx.y * TILE + ty
    for j in range(CO_PER_THREAD):
        acc[j] = float32(0.0)
    for c0 in range(0, c_in, CI_CHUNK):
        for idx in range(tid, HALO * HALO * CI_CHUNK, TILE * TILE):      # carga cooperativa
            ci = idx % CI_CHUNK
            p = idx // CI_CHUNK
            gy = cuda.blockIdx.y * TILE + p // HALO - 1
            gx = cuda.blockIdx.x * TILE + p % HALO - 1
            value = float32(0.0)
            if 0 <= gy < height and 0 <= gx < width and c0 + ci < c_in:
                value = x[n, gy, gx, c0 + ci]
            tile[p // HALO, p % HALO, ci] = value
        for idx in range(tid, 9 * CI_CHUNK * CO_PER_THREAD, TILE * TILE):
            co = idx % CO_PER_THREAD
            r = idx // CO_PER_THREAD
            ci = r % CI_CHUNK
            kk = r // CI_CHUNK
            value = float32(0.0)
            if c0 + ci < c_in and co0 + co < c_out:
                value = w[kk // 3, kk % 3, c0 + ci, co0 + co]
            sw[kk // 3, kk % 3, ci, co] = value
        cuda.syncthreads()
        for ky in range(3):
            for kx in range(3):
                for ci in range(CI_CHUNK):
                    xv = tile[ty + ky, tx + kx, ci]
                    for j in range(CO_PER_THREAD):
                        acc[j] += xv * sw[ky, kx, ci, j]
        cuda.syncthreads()                    # nadie sobrescribe la tesela mientras otro la lee
    if ox < width and oy < height:
        for j in range(CO_PER_THREAD):
            co = co0 + j
            if co < c_out:
                value = acc[j] + b[co]
                if relu and value < 0:
                    value = float32(0.0)
                out[n, oy, ox, co] = value


def conv_tiled_config(width: int, height: int, images: int, filters: int):
    groups = (filters + CO_PER_THREAD - 1) // CO_PER_THREAD
    return ((width + TILE - 1) // TILE, (height + TILE - 1) // TILE, images * groups), (TILE, TILE)


@cuda.jit("void(float32[:, :, :, :], float32[:, :, :, :])")
def maxpool_2x2(x, out):
    """MaxPooling 2×2 con stride 2. Un hilo por valor de salida; blockIdx.z = (imagen, canal)."""
    ox, oy = cuda.grid(2)
    z = cuda.blockIdx.z
    channels = out.shape[3]
    n = z // channels
    c = z % channels
    if ox >= out.shape[2] or oy >= out.shape[1]:
        return
    iy, ix = 2 * oy, 2 * ox
    best = x[n, iy, ix, c]
    for dy in range(2):
        for dx in range(2):
            value = x[n, iy + dy, ix + dx, c]
            if value > best:
                best = value
    out[n, oy, ox, c] = best


@cuda.jit("void(float32[:, :], float32[:, :], float32[:], float32[:, :], int32)")
def matmul_bias_act(a, bmat, bias, c, relu):
    """C = A·B + sesgo (y ReLU opcional) con teselas de 16 × 16 en memoria compartida.

    Cada bloque calcula una tesela 16 × 16 de C. En cada iteración los 256 hilos copian
    juntos una tesela de A y una de B a memoria compartida, se sincronizan y cada hilo
    acumula 16 productos. Así cada valor de A y B se lee de memoria global 16 veces menos.
    """
    sa = cuda.shared.array(shape=(TILE, TILE), dtype=float32)
    sb = cuda.shared.array(shape=(TILE, TILE), dtype=float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    col = cuda.blockIdx.x * TILE + tx
    row = cuda.blockIdx.y * TILE + ty
    rows, inner, cols = a.shape[0], a.shape[1], bmat.shape[1]
    acc = float32(0.0)
    for t in range((inner + TILE - 1) // TILE):
        k_a = t * TILE + tx
        k_b = t * TILE + ty
        sa[ty, tx] = a[row, k_a] if row < rows and k_a < inner else float32(0.0)
        sb[ty, tx] = bmat[k_b, col] if k_b < inner and col < cols else float32(0.0)
        cuda.syncthreads()
        for k in range(TILE):
            acc += sa[ty, k] * sb[k, tx]
        cuda.syncthreads()
    if row < rows and col < cols:
        acc += bias[col]
        if relu and acc < 0:
            acc = float32(0.0)
        c[row, col] = acc


@cuda.jit("void(float32[:, :], float32[:, :])")
def softmax_rows(z, out):
    """Un hilo calcula la softmax de una fila (estable: resta el máximo antes de exp)."""
    r = cuda.grid(1)
    if r >= z.shape[0]:
        return
    top = z[r, 0]
    for j in range(1, z.shape[1]):
        if z[r, j] > top:
            top = z[r, j]
    total = float32(0.0)
    for j in range(z.shape[1]):
        total += math.exp(z[r, j] - top)
    for j in range(z.shape[1]):
        out[r, j] = math.exp(z[r, j] - top) / total


# ------------------------------------------------------------------ configuraciones
def grid_2d(width: int, height: int, depth: int) -> tuple[tuple[int, int, int], tuple[int, int]]:
    return ((width + TILE - 1) // TILE, (height + TILE - 1) // TILE, depth), (TILE, TILE)


def matmul_config(rows: int, cols: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return ((cols + TILE - 1) // TILE, (rows + TILE - 1) // TILE), (TILE, TILE)


def softmax_config(rows: int) -> tuple[int, int]:
    return (rows + SOFTMAX_THREADS - 1) // SOFTMAX_THREADS, SOFTMAX_THREADS


class NumbaCNN:
    """Forward completo de la CNN base con los kernels propios. Pesos en formato Keras."""

    def __init__(self, keras_weights: list, conv: str = "tiled"):
        if conv not in ("tiled", "naive"):
            raise ValueError(conv)
        self.conv_version = conv
        w = [cuda.to_device(array.astype("float32")) for array in keras_weights]
        self.conv = [(w[0], w[1]), (w[2], w[3]), (w[4], w[5])]
        self.dense = [(w[6], w[7], 1), (w[8], w[9], 0)]
        self.launches: list[dict] = []

    def _record(self, name, grid, block):
        grid = grid if isinstance(grid, tuple) else (grid,)
        block = block if isinstance(block, tuple) else (block,)
        blocks = math.prod(grid)
        threads = math.prod(block)
        self.launches.append({"kernel": name, "grid": "x".join(map(str, grid)),
                              "block": "x".join(map(str, block)), "blocks": blocks,
                              "threads_per_block": threads, "total_threads": blocks * threads})

    def forward(self, x_device, record: bool = False, stages: dict | None = None):
        """x_device: (N, 64, 64, 3) en GPU. Devuelve probabilidades (N, 29) en GPU."""
        if record:
            self.launches = []
        h = x_device
        n = h.shape[0]
        for index, (kernel, bias) in enumerate(self.conv, start=1):
            height, width, filters = h.shape[1], h.shape[2], kernel.shape[3]
            conv_out = cuda.device_array((n, height, width, filters), dtype="float32")
            if self.conv_version == "tiled":
                grid, block = conv_tiled_config(width, height, n, filters)
                conv2d_3x3_tiled[grid, block](h, kernel, bias, conv_out, 1)
                kernel_name = "conv2d_3x3_tiled"
            else:
                grid, block = grid_2d(width, height, n * filters)
                conv2d_3x3_bias_relu[grid, block](h, kernel, bias, conv_out, 1)
                kernel_name = "conv2d_3x3_bias_relu"
            if record:
                self._record(f"{kernel_name} (capa {index})", grid, block)
            pooled = cuda.device_array((n, height // 2, width // 2, filters), dtype="float32")
            grid, block = grid_2d(width // 2, height // 2, n * filters)
            maxpool_2x2[grid, block](conv_out, pooled)
            if record:
                self._record(f"maxpool_2x2 (capa {index})", grid, block)
            h = pooled
            if stages is not None:
                stages[f"block{index}"] = h
        h = h.reshape((n, h.shape[1] * h.shape[2] * h.shape[3]))   # Flatten NHWC, sin copia
        for index, (kernel, bias, relu) in enumerate(self.dense, start=1):
            out = cuda.device_array((n, kernel.shape[1]), dtype="float32")
            grid, block = matmul_config(n, kernel.shape[1])
            matmul_bias_act[grid, block](h, kernel, bias, out, relu)
            if record:
                self._record(f"matmul_bias_act (Dense {index})", grid, block)
            h = out
        probabilities = cuda.device_array_like(h)
        grid, block = softmax_config(n)
        softmax_rows[grid, block](h, probabilities)
        if record:
            self._record("softmax_rows", grid, block)
        return probabilities


__all__ = ["NUMBA_SETUP", "NumbaCNN", "adam_launch_config", "adam_update", "conv2d_3x3_bias_relu",
           "conv2d_3x3_tiled", "conv_tiled_config",
           "cuda", "int32", "matmul_bias_act", "maxpool_2x2", "softmax_rows"]
