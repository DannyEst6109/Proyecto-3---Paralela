"""La red de referencia debe conservar la arquitectura de la CNN original."""
import ast
import json
import os
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from asl_cpu.common import ROOT, load_config
from asl_cpu.model import create_cnn_base


class ModelTests(unittest.TestCase):
    def test_model_matches_original_cnn_base(self):
        import tensorflow as tf
        config = load_config(ROOT / "config/baseline.json")
        notebook = json.loads((ROOT / "reference/Laboratorio3.original.ipynb").read_text(encoding="utf-8"))
        tree = ast.parse("".join(notebook["cells"][59]["source"]))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "crear_cnn_base")
        namespace = {"tf": tf, "keras": tf.keras, "layers": tf.keras.layers,
                     "SEMILLA": 42, "FORMA": (64, 64, 3), "NUM_CLASES": 29}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "original_cnn", "exec"), namespace)
        original = namespace["crear_cnn_base"]()
        adapted = create_cnn_base(tf, config)
        self.assertEqual(adapted.count_params(), original.count_params())
        self.assertEqual(adapted.count_params(), 1145693)
        self.assertEqual(len(adapted.layers), len(original.layers))
        for left, right in zip(original.layers, adapted.layers):
            self.assertEqual(type(left), type(right))
            left_config, right_config = left.get_config(), right.get_config()
            # Keras genera nombres distintos al construir dos modelos en un proceso.
            left_config.pop("name")
            right_config.pop("name")
            self.assertEqual(left_config, right_config)
        self.assertEqual([tuple(w.shape) for w in original.weights], [tuple(w.shape) for w in adapted.weights])


if __name__ == "__main__":
    unittest.main()
