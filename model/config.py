import torch
from dataclasses import dataclass


@dataclass
class ModelConfig:
    weights_dtype: torch.dtype = torch.bfloat16
    vocab_size: int = 262_144
    max_position_embeddings: int = 32_768
    num_hidden_layers: int = 18
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    hidden_size: int = 640
    intermediate_size: int = 2048
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    quant: bool = False
    tokenizer: str = "tokenizer/tokenizer.model"
    attn_types: tuple = (
        "local_sliding",
        "local_sliding",
        "local_sliding",
        "local_sliding",
        "local_sliding",
        "global",
    )
    sliding_window_size: int = 512
    query_pre_attn_scalar: int = 256
    use_pre_ffw_norm: bool = True
    use_post_ffw_norm: bool = True
    rope_local_base_freq: int = 10_000
    rope_theta: int = 1_000_000
    rope_wave_length = {"local_sliding": 10_000, "global": 1_000_000}
    use_qk_norm: bool = True
