import fire
import jax
import jax.numpy as jnp
import optax

from nnx_benchmark.models.ViT.model import VisionTransformer

def train_step(
        
):
    pass



def train(
    num_epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
):
    pass    

if __name__ == '__main__':
   fire.Fire(train)