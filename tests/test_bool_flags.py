import unittest

from expedantic import ConfigBase


class Config(ConfigBase):
    class Inner(ConfigBase):
        cold: bool = True

    inner: Inner = Inner()
    verbose: bool = False


class TestFlexibleBooleans(unittest.TestCase):
    def parse(self, args):
        return Config.parse_args(args=args, diff_print_mode="none", print_config=False)

    def test_bare_flag_sets_true(self):
        self.assertTrue(self.parse(["--verbose"]).verbose)

    def test_no_prefix_sets_false(self):
        self.assertFalse(self.parse(["--no-inner.cold"]).inner.cold)

    def test_explicit_word_values(self):
        self.assertFalse(self.parse(["--inner.cold", "false"]).inner.cold)
        self.assertTrue(self.parse(["--verbose", "true"]).verbose)
        self.assertFalse(self.parse(["--inner.cold", "False"]).inner.cold)

    def test_numeric_and_yesno_values(self):
        self.assertFalse(self.parse(["--inner.cold", "0"]).inner.cold)
        self.assertTrue(self.parse(["--verbose", "1"]).verbose)
        self.assertFalse(self.parse(["--inner.cold", "no"]).inner.cold)
        self.assertTrue(self.parse(["--verbose", "yes"]).verbose)

    def test_equals_form(self):
        self.assertFalse(self.parse(["--inner.cold=false"]).inner.cold)
        self.assertTrue(self.parse(["--verbose=true"]).verbose)

    def test_garbage_value_errors(self):
        with self.assertRaises(SystemExit):
            self.parse(["--inner.cold", "maybe"])

    def test_no_prefix_rejects_value(self):
        with self.assertRaises(SystemExit):
            self.parse(["--no-verbose", "true"])

    def test_defaults_untouched(self):
        cfg = self.parse([])
        self.assertTrue(cfg.inner.cold)
        self.assertFalse(cfg.verbose)


if __name__ == "__main__":
    unittest.main()
