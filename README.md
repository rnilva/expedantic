# Expedantic

[![pypi](https://img.shields.io/pypi/v/expedantic.svg)](https://pypi.python.org/pypi/expedantic)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

## Installation

```
pip install expedantic
```

## Basic Usage
```python
from expedantic import ConfigBase

# Define a config model
class MyConfig(ConfigBase):
    device: str = 'cuda:0'
    learning_rate: float = 1.0e-3
    num_epochs: int = 100

# Save and load from yaml files
my_config = MyConfig()
my_config.save_as_yaml("config.yaml")
my_config = MyConfig.load_from_yaml("config.yaml")


def learn(device: str, learning_rate: float, num_epochs: int):
    ...

# Consume the config by getting kwargs automatically.
learn(**my_config.compatible_args(learn))


# Or pass manually
learn(my_config.device, my_config.learning_rate, my_config.num_epochs)
```


## Features

- Type validation using `pydantic`.

- JSON schema generation for autocompletion on yaml files.

- Integrated argument parser with supporting nested key access:
    ```python
    # run.py
    class MyInnerConfig(ConfigBase):
        inner_key: str = "inner_value"
    class MyConfig(ConfigBase):
        inner_config: MyInnerConfig = MyInnerConfig()
        outer_key: int = 10


    my_config = MyConfig.parse_args()
    ```
    ```shell
    python run.py --inner_config.inner_key "another inner value" --outer_key 20
    ```

- `!include` directive support for yaml files:
    ```yaml
    # base.yaml
    learning_rate: 3.0e-5
    num_epochs: 10
    device: cpu
    ```
    ```yaml
    # derived.yaml
    <<: !include base.yaml
    device: cuda
    extra: True
    ```

    The content of `derived.yaml` is equivalent to the following:
    ```yaml
    learning_rate: 3.0e-5
    num_epochs: 10
    device: cuda
    extra: True
    ```

- Mutually exclusive configuration groups:

    ```python
    from pydantic import ValidationError


    class Config(ConfigBase):
        _mutually_exclusive_sets = [{"make_algorithm_A_obsolete", "use_algorithm_A"}]

        make_algorithm_A_obsolete: bool = True
        use_algorithm_A: bool = False

    try:
        config = Config(make_algorithm_A_obsolete=True, use_algorithm_A=True)
    except ValidationError:
        print(e)
        """
        1 validation error for Config
            Value error, Mutual exclusivity has broken. (set: {'a', 'b'}) [type=value_error, input_value={'make_algorithm_A_obsolete': True, 'use_algorithm_A': True}, input_type=dict]
        """

    ```


