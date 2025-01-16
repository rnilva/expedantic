from ruamel.yaml import YAML as RUAMEL_YAML
from ccorp.ruamel.yaml.include import YAML
import pydantic_yaml

pydantic_yaml._internals.v2.YAML = YAML

yaml = YAML()
yaml.sort_base_mapping_type_on_output = False
yaml.default_flow_style = False
yaml.indent(mapping=2, sequence=4, offset=2)

yaml.representer.add_representer(
    type(None),
    lambda self, _: self.represent_scalar("tag:yaml.org,2002:null", "null"),
)
