from setuptools import setup, find_packages


setup(
    name="expedantic",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pydantic >= 2.6.3",
        "pydantic_yaml >= 1.2.1",
        "pyyaml",
        "typed-argument-parser",
    ],
    author="rnilva",
    author_email="rnilva849@gmail.com",
    description="",
    license="MIT",
    keywords="",
    url="https://github.com/rnilva/expedantic",
)
