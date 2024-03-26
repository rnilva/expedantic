# https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions
from pathlib import Path
from typing import Literal

from expedantic import ConfigBase, Field


class GANConfig(ConfigBase):
    name: Literal["GAN"] = "GAN"
    learning_rate_generator: float = 1.0e-4
    learning_rate_discriminator: float = 1.0e-5
    batch_size: int = 128
    n_epochs: int = 10
    latent_size: int = 16


class WGANConfig(GANConfig):
    name: Literal["WGAN"] = "WGAN"
    weight_clipping_range: float = 0.01
    n_critic_epochs: int = 5


class WGANGPConfig(WGANConfig):
    name: Literal["WGAN_GP"] = "WGAN_GP"
    gp_lambda: float = 10


class Config(ConfigBase):
    device: Literal["cpu", "cuda"] = "cpu"
    dataset: Literal["mnist", "cifar10"] = "mnist"
    checkpoint_freq: int = 1000
    checkpoint_save_dir: Path = Path("./checkpoints/")
    verbose: bool = True
    gan_config: GANConfig | WGANConfig | WGANGPConfig = Field(..., discriminator="name")


if __name__ == "__main__":
    config = Config(gan_config={"name": "WGAN"})  # type: ignore
    print(config.gan_config)
    """
    name='WGAN' learning_rate_generator=0.0001 
    learning_rate_discriminator=1e-05 batch_size=128
    n_epochs=10 latent_size=16 weight_clipping_range=0.01
    n_critic_epochs=5
    """
