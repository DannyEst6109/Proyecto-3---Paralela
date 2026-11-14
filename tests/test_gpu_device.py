import unittest
from unittest.mock import Mock

from asl_cpu.profile_kernels import classify
from asl_cpu.train import resolve_training_device


class TrainingDeviceSelectionTests(unittest.TestCase):
    def test_auto_selects_cpu_when_no_gpu_is_visible(self):
        tf = Mock()
        tf.config.list_physical_devices.return_value = []
        self.assertEqual(resolve_training_device(tf, "auto"), "/CPU:0")

    def test_gpu_requested_requires_a_cuda_device(self):
        tf = Mock()
        tf.config.list_physical_devices.return_value = []
        with self.assertRaisesRegex(RuntimeError, "No CUDA-capable GPU"):
            resolve_training_device(tf, "gpu")

    def test_gpu_selection_reports_the_active_device(self):
        tf = Mock()
        gpu = Mock()
        tf.config.list_physical_devices.return_value = [gpu]
        tf.config.get_visible_devices.return_value = [gpu]
        self.assertEqual(resolve_training_device(tf, "gpu"), "/GPU:0")

    def test_cpu_mode_never_queries_gpus(self):
        tf = Mock()
        self.assertEqual(resolve_training_device(tf, "cpu"), "/CPU:0")
        tf.config.list_physical_devices.assert_not_called()


class KernelClassificationTests(unittest.TestCase):
    def test_kernels_are_grouped_by_their_tensorflow_op(self):
        self.assertEqual(classify("ampere_sgemm_128x128_nt", ["Conv2DBackpropFilter"]),
                         "Conv2D gradiente de filtros (cuDNN)")
        self.assertEqual(classify("winograd", ["_FusedConv2D"]),
                         "Conv2D forward (cuDNN, fusionada con sesgo+ReLU)")
        self.assertEqual(classify("void Eigen::internal::EigenMetaKernel<...>", ["ReluGrad"]),
                         "Gradiente ReLU (TF/Eigen)")
        self.assertEqual(classify("ampere_sgemm_32x32", ["MatMul"]), "Dense: producto de matrices (cuBLAS)")


if __name__ == "__main__":
    unittest.main()
