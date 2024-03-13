import unittest
from pathlib import Path
from pprint import pprint

from pydantic import RootModel, BaseModel
from pydantic_yaml import parse_yaml_file_as

from ccorp.ruamel.yaml.include import YAML
from expedantic import ConfigBase, Field


class Bear(BaseModel):
    kingdom: str
    phylum: str
    class_: str = Field(alias="class")
    order: str
    subordia: str
    superfamily: str
    family: str


class ExtendedBear(Bear):
    color: str
    size: str
    scary: bool


class Bears(RootModel[dict[str, Bear]]): ...


class ExtendedBears(RootModel[dict[str, ExtendedBear]]): ...


class Base(ConfigBase):
    device: str = "cpu"
    learning_rate: float = 3.0e-4
    batch_size: int = 1024


class Child(Base):
    gradient_clip_range: float = 0.2


class GrandChild(Child):
    decoder_net_arch: list[int] = [128, 128]


class TestInclude(unittest.TestCase):
    def setUp(self) -> None:
        base = """\
device: cpu
batch_size: 1024
learning_rate: 0.0003
"""
        child = """\
<<: !include ./tmp/base.yaml
device: 'cuda'
gradient_clip_range: 0.2
"""
        grandchild = """\
<<: !include ./tmp/child.yaml
learning_rate: 5.0e-5
gradient_clip_range: 0.3
decoder_net_arch:
- 128
- 128
"""
        yaml = YAML()

        self.base_file = Path("./tmp/base.yaml")
        self.child_file = Path("./tmp/child.yaml")
        self.grand_child_file = Path("./tmp/grand_child.yaml")

        self.base_file.parent.mkdir(exist_ok=True)
        yaml.dump(yaml.load(base), self.base_file.open("w"))
        yaml.dump(yaml.load(child), self.child_file.open("w"))
        yaml.dump(yaml.load(grandchild), self.grand_child_file.open("w"))
        ...

    def tearDown(self) -> None:
        self.base_file.unlink()
        self.child_file.unlink()
        self.grand_child_file.unlink()

    def test_include(self):
        config = Base.load_from_yaml(self.base_file)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.batch_size, 1024)
        self.assertEqual(config.learning_rate, 3.0e-4)

        config = Child.load_from_yaml(self.child_file)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.gradient_clip_range, 0.2)

        config = GrandChild.load_from_yaml(self.grand_child_file)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.learning_rate, 5.0e-5)
        self.assertEqual(config.gradient_clip_range, 0.3)
        self.assertEqual(config.decoder_net_arch, [128, 128])


if __name__ == "__main__":
    unittest.main()

    # c = GrandChild()
    # c.save_as_yaml("test_include.yaml")

    # yaml = YAML(pure=True)
    # with open("tests/yaml/data/root.yaml", "r") as f:
    #     data = yaml.load(f)
    # pprint(data)

    # Bear.model_validate(data[''])
