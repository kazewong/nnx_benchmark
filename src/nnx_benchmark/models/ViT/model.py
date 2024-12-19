import jax
from flax import nnx
from jaxtyping import Int, Float, Array, PRNGKeyArray


class AttentionBlock(nnx.Module):
    """
    LayerNorm1 -> Attention -> Dropout -> Residual -> LayerNorm2 -> FeedForward -> Dropout -> Residual
    """

    def __init__(
        self,
        rng_seed: Int,
        embed_dim: Int,
        hidden_dim: Int,
        qkv_dim: Int,
        num_heads: Int,
        dropout_rate: Float = 0.0,
    ):
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=qkv_dim,
            out_features=embed_dim,
            dropout_rate=dropout_rate,
            rngs=nnx.Rngs(rng_seed),
        )
        self.layer_norm1 = nnx.LayerNorm(
            num_features=embed_dim, rngs=nnx.Rngs(rng_seed + 1)
        )
        self.layer_norm2 = nnx.LayerNorm(
            num_features=embed_dim, rngs=nnx.Rngs(rng_seed + 2)
        )
        self.dropout = nnx.Dropout(dropout_rate, rngs=nnx.Rngs(rng_seed + 3))
        self.linear_block 

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)


class VisionTransformer(nnx.Module):

    def __init__(
        self,
        rng_seed: int,
        embed_dim: Int,  # Dimensionality of input and attention feature vectors
        hidden_dim: int,  # Dimensionality of hidden layer in feed-forward network
        num_heads: int,  # Number of heads to use in the Multi-Head Attention block
        num_channels: int,  # Number of channels of the input (3 for RGB)
        num_layers: int,  # Number of layers to use in the Transformer
        num_classes: int,  # Number of classes to predict
        patch_size: int,  # Number of pixels that the patches have per dimension
        num_patches: int,  # Maximum number of patches an image can have
        dropout_prob: float = 0.0,  # Amount of dropout to apply in the feed-forward network
    ):
        self.patch_size = patch_size

        self.linear_projector = nnx.Linear(
            patch_size * patch_size * num_channels, embed_dim, rngs=nnx.Rngs(rng_seed)
        )
        self.attention_blocks = [
            AttentionBlock(rng_seed, embed_dim, hidden_dim, num_heads)
            for i in range(num_layers)
        ]
        self.dropout_block = nnx.Dropout(dropout_prob, rngs=rng)
        self.feedforward_head
        self.positional_embedding = nnx.E

    def __call__(
        self, images: Int[Array, "N C H W"], inference: bool = False
    ) -> Int[Array, "N C"]:
        x = self.image_to_patch(images)
        x = nnx.vmap(self.linear_projector)(x)
        x = x + self.positional_embedding(x)
        x = self.dropout_block(x)

    def image_to_patch(
        self, image: Int[Array, " C H W"]
    ) -> Int[Array, "N H*W/P/P P*P*C"]:
        raise NotImplementedError
