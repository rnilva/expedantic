# Expedantic

![pypi](imgs/pypi.svg)
![versions](imgs/python.svg)

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

- Integrated argument parser with supporting nested key accessment:
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

- `!include` derivative support for yaml files.

