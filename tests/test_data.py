"""Regresiones de equivalencia con el laboratorio y de integridad de las imágenes."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from asl_cpu.common import EXPECTED_CLASSES, ROOT, load_config
from asl_cpu.data import prepare_arrays, read_image, select_partitions


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "config/baseline.json")
        cls.inventory = pd.DataFrame([
            {"clase": label, "indice": i, "archivo": f"{label}{i}.jpg",
             "relative_path": f"{label}/{label}{i}.jpg"}
            for label in EXPECTED_CLASSES for i in range(1, 3001)
        ])

    def test_matches_original_notebook_partition(self):
        notebook = json.loads((ROOT / "reference/Laboratorio3.original.ipynb").read_text(encoding="utf-8"))
        namespace = {"np": np, "pd": pd, "SEMILLA": 42, "CLASES": EXPECTED_CLASSES,
                     "inventario": self.inventory.copy()}
        with contextlib.redirect_stdout(io.StringIO()):
            for index in (34, 35):
                exec("".join(notebook["cells"][index]["source"]), namespace)
        _, actual = select_partitions(self.inventory, self.config)
        pd.testing.assert_frame_equal(actual, namespace["particion"])

    def test_counts_unique_files_and_block_guards(self):
        _, selected = select_partitions(self.inventory, self.config)
        self.assertEqual(selected["conjunto"].value_counts().to_dict(), {"train": 12180, "val": 2610, "test": 2610})
        self.assertFalse(selected["relative_path"].duplicated().any())
        self.assertTrue(selected["posicion"].between(25, 74).all())
        self.assertEqual(selected.groupby(["clase", "bloque"])["conjunto"].nunique().max(), 1)

    def test_rejects_more_samples_than_available(self):
        config = dict(self.config, samples_per_class={"train": 2000, "val": 90, "test": 90})
        with self.assertRaisesRegex(ValueError, "suficientes"):
            select_partitions(self.inventory, config)

    def test_rejects_unreadable_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corrupt.jpg"
            path.write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "No se pudo leer"):
                read_image((str(path), 64))

    def test_process_pool_preserves_pixels_labels_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for split in ("train", "val", "test"):
                for label in ("A", "space"):
                    for index in range(3):
                        relative = f"{split}_{label}{index}.png"
                        pixels = np.random.default_rng(index + len(rows)).integers(0, 256, (80, 80, 3), dtype=np.uint8)
                        self.assertTrue(cv2.imwrite(str(root / relative), pixels))
                        rows.append({"clase": label, "conjunto": split, "relative_path": relative})
            partition = pd.DataFrame(rows)
            with contextlib.redirect_stdout(io.StringIO()):
                sequential = prepare_arrays(partition, root, 64, 1)
                parallel = prepare_arrays(partition, root, 64, 2)
            for key in sequential:
                np.testing.assert_array_equal(sequential[key], parallel[key])
            # La etiqueta space ocupa el índice 28 del mapa común, no el 1 de esta muestra.
            np.testing.assert_array_equal(sequential["y_train"], [0, 0, 0, 28, 28, 28])
            original = cv2.imread(str(root / rows[0]["relative_path"]))
            expected = cv2.cvtColor(cv2.resize(original, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(sequential["X_train"][0], expected)


if __name__ == "__main__":
    unittest.main()
