import unittest

from pydantic import ValidationError

from expedantic import ConfigBase


class Config(ConfigBase):
    _mutually_exclusive_sets = [{"use_mlp_model", "use_batch_norm_on_cnns"}]

    use_mlp_model: bool = True
    use_batch_norm_on_cnns: bool = False


class TestMutuallyExclusiveSets(unittest.TestCase):
    def test_mutually_exclusive_bools(self):
        with self.assertRaises(ValidationError):
            config = Config(use_mlp_model=True, use_batch_norm_on_cnns=True)

        with self.assertRaises(ValidationError):
            config = Config(use_mlp_model=False, use_batch_norm_on_cnns=False)

        config = Config(use_mlp_model=False, use_batch_norm_on_cnns=True)
        self.assertEqual(config.use_mlp_model, False)
        self.assertEqual(config.use_batch_norm_on_cnns, True)
