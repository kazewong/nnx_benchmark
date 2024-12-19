import flax.nnx as nnx
import jax.numpy as jnp
import jax

from nnx_benchmark.models.ViT.model import VisionTransformer

model = VisionTransformer(
    rng_seed=0,
    embed_dim=16,
    hidden_dim=64,
    num_heads=4,
    num_channels=3,
    num_layers=4,
    patch_size=4,
    num_patches=64,
)

test_input = jnp.zeros((10, 3, 16, 16))

def single_eval(input):
    return model(input)

nnx.jit(nnx.vmap(single_eval))(test_input)