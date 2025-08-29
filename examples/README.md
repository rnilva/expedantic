# Examples

Here, we provide a few example usages of `expedantic` features, especially ones that are based on [`pydantic`](https://docs.pydantic.dev/latest/) features.

- [`discriminated_unions.py`](discriminated_unions.py) demonstrates the use of 'discriminated unions' feature of `pydantic`. This feature is handy when a field can be one of multiple composite types.

- [`logger_usage.py`](logger_usage.py) demonstrates comprehensive usage of the `expedantic.logger` module, including:
  - All 8 field types with different aggregation strategies
  - Realistic ML training loop logging
  - Custom domain-specific loggers
  - Data export and DataFrame conversion

