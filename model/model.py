import gc
import os
from typing import Literal, Union, Mapping
import torch
from torch import nn
import torch.nn.functional as F
from config import ModelConfig
from tokenizer import Tokenizer


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, rope_scaling_factor: int = 1) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = freqs / rope_scaling_factor
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(1, 2)
    return x_out


class GatedMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor):
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, attn_type: Literal["local_sliding", "global"]):
        super().__init__()
        if attn_type not in ["local_sliding", "global"]:
            raise ValueError(f"attn_type must be 'local_sliding' or 'global', got '{attn_type}'")

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = config.query_pre_attn_scalar**-0.5

        self.qkv_proj = nn.Linear(self.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        self.query_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_type = attn_type
        self.sliding_window_size = config.sliding_window_size
        self.attn_types = config.attn_types

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        if self.query_norm is not None and self.key_norm is not None:
            xq = self.query_norm(xq)
            xk = self.key_norm(xk)

        # Positional embedding.
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        q.mul_(self.scaling)
        scores = torch.matmul(q, k.transpose(2, 3))
        if self.attn_type == "local_sliding" and self.sliding_window_size is not None and local_mask is not None:
            mask = local_mask

        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)

        # [batch_size, input_len, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
        output = self.o_proj(output)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, attn_type: Literal["local_sliding", "global"]):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = Attention(
            config=config,
            attn_type=self.attn_type,
        )
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = (
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_pre_ffw_norm else None
        )
        self.post_feedforward_layernorm = (
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_post_ffw_norm else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
            local_mask=local_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class BaseModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            attn_type = config.attn_types[i % len(config.attn_types)] if config.attn_types is not None else "global"
            self.layers.append(DecoderLayer(config, attn_type))

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Mapping[Literal["local_sliding", "global"], torch.Tensor],
        kv_write_indices: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis.get(layer.attn_type),
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
                local_mask=local_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class CausalLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0

        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        self.tokenizer = Tokenizer("tokenizer/tokenizer.model")
        self.embedder = nn.Embedding(vocab_size, config.hidden_size)
        self.model = BaseModel(config)

        rope_lengths = config.rope_wave_length
        defaults = {"local_sliding": 10_000, "global": 1_000_000}

        for attn_type, name in [
            ("local_sliding", "local_freqs_cis"),
            ("global", "global_freqs_cis"),
        ]:
            theta = rope_lengths.get(attn_type, defaults[attn_type])

            self._register_freqs_cis(name, head_dim, max_seq_len, theta=theta)

        else:
            self._register_freqs_cis("freqs_cis", head_dim, max_seq_len)

    def _register_freqs_cis(self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000):
        self.register_buffer(name, precompute_freqs_cis(head_dim, max_seq_len * 2, theta=theta))

    @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        local_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        freqs_cis: dict = {}
        freqs_cis["local_sliding"] = self.local_freqs_cis.index_select(0, input_positions)
        freqs_cis["global"] = self.global_freqs_cis.index_select(0, input_positions)

        kv_write_indices = input_positions
        # [batch_size, input_len, hidden_size]
        hidden_states = self.embedder(input_token_ids)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=local_mask,
        )
        embedder_weight = self.embedder.weight

        return embedder_weight, hidden_states, output_positions, temperatures, top_ps, top_ks
