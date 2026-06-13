import io
import unittest
import tempfile
from pathlib import Path
from contextlib import redirect_stdout

from expedantic import ConfigBase
from expedantic.utils import compose_layers, update_recursively


class Config(ConfigBase):
    class Train(ConfigBase):
        lrm: float = 0.5
        iters: int = 2400

    tag: str = "run"
    train: Train = Train()


class TestLayerComposition(unittest.TestCase):
    def test_compose_layers_deep_merge(self):
        base = {"a": 1, "nested": {"x": 1, "y": 2}}
        layer = {"nested": {"y": 20}, "b": 3}
        out = compose_layers(base, layer)
        self.assertEqual(out, {"a": 1, "nested": {"x": 1, "y": 20}, "b": 3})
        # inputs untouched
        self.assertEqual(base, {"a": 1, "nested": {"x": 1, "y": 2}})
        self.assertEqual(layer, {"nested": {"y": 20}, "b": 3})

    def test_update_recursively_in_place(self):
        d = {"nested": {"x": 1}}
        update_recursively(d, {"nested": {"y": 2}})
        self.assertEqual(d, {"nested": {"x": 1, "y": 2}})


class TestLayeredParse(unittest.TestCase):
    def _yaml(self, content: str) -> Path:
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        f.write(content)
        f.close()
        return Path(f.name)

    def test_file_plus_cli_values(self):
        path = self._yaml("tag: cell_a\ntrain:\n  lrm: 0.25\n")
        cfg = Config.parse_args(
            require_default_file=True,
            args=[str(path), "--train.iters", "1200"],
            diff_print_mode="none",
            print_config=False,
        )
        self.assertEqual(cfg.tag, "cell_a")        # from the file
        self.assertEqual(cfg.train.lrm, 0.25)      # from the file
        self.assertEqual(cfg.train.iters, 1200)    # CLI override

    def test_cli_wins_over_file(self):
        path = self._yaml("train:\n  lrm: 0.25\n")
        cfg = Config.parse_args(
            require_default_file=True,
            args=[str(path), "--train.lrm", "0.125"],
            diff_print_mode="none",
            print_config=False,
        )
        self.assertEqual(cfg.train.lrm, 0.125)

    def test_layered_diff_prints_both_stages(self):
        path = self._yaml("train:\n  lrm: 0.25\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            Config.parse_args(
                require_default_file=True,
                args=[str(path), "--train.iters", "1200"],
                diff_print_mode="tree",
                print_config=False,
            )
        out = buf.getvalue()
        self.assertIn("config file", out)
        self.assertIn("CLI overrides", out)

    def test_no_file_single_stage(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            Config.parse_args(
                args=["--train.lrm", "0.25"],
                diff_print_mode="tree",
                print_config=False,
            )
        out = buf.getvalue()
        self.assertNotIn("CLI overrides", out)     # single-stage legacy rendering


if __name__ == "__main__":
    unittest.main()
