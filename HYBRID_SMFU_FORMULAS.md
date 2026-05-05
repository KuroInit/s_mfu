# Hybrid S-MFU and S-MBU Formulas for Qwen3-Next and DeepSeek V4

This note proposes an architecture-aware extension of S-MFU/S-MBU for hybrid MoE models. The goal is to keep the metric faithful to MoE-CAP's intent: count the work and bytes actually activated by the architecture, not the dense-equivalent cost of every expert.

References:

- [Qwen3-Next-80B-A3B-Instruct config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json)
- [DeepSeek-V4-Flash config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)
- [DeepSeek-V4-Pro-Base config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base/blob/main/config.json)

## Core Definition

For prefill, define:

```text
F_total = sum of architecture-activated FLOPs for the request
B_total = sum of architecture-accessed bytes for the request
T       = measured prefill latency, normally TTFT
P_peak  = hardware peak FLOPs/s
BW_peak = hardware peak memory bandwidth in bytes/s
```

Then:

```text
achieved_FLOPs_per_s = F_total / T
achieved_bytes_per_s = B_total / T

hybrid_S_MFU = achieved_FLOPs_per_s / P_peak
hybrid_S_MBU = achieved_bytes_per_s / BW_peak

arithmetic_intensity = F_total / B_total
```

For a roofline plot:

```text
x = arithmetic_intensity
y = achieved_FLOPs_per_s

compute_roof = P_peak
memory_roof(x) = BW_peak * x
```

MFU and MBU are not separate axes. They are readings relative to the compute and memory roofs:

```text
S_MFU = y / compute_roof
S_MBU = (y / x) / BW_peak
```

## Generic Layer Sum

Compute the numerator per layer and sum:

```text
F_total =
    F_attention_full
  + F_attention_linear
  + F_attention_sliding
  + F_attention_compressed_sparse
  + F_dense_mlp
  + F_routed_experts
  + F_shared_experts
  + F_router
  + F_projection_norm_misc

B_total =
    B_attention_weights
  + B_attention_cache_or_state
  + B_dense_mlp_weights
  + B_routed_expert_weights
  + B_shared_expert_weights
  + B_router_weights
  + B_kv_or_compressed_cache
  + B_projection_norm_misc
```

The metric remains sparse because routed experts are multiplied by active experts per token, not by total experts.

## Shared Building Blocks

Let:

```text
B = batch size
S = prefill input sequence length
H = hidden size
I = dense MLP intermediate size
E = MoE expert intermediate size
K = active routed experts per token
G = shared experts per token
L = number of layers of this type
dtype_bytes = bytes per parameter or activation element
```

For SwiGLU-style FFNs, use:

```text
F_dense_mlp_layer = 3 * 2 * B * S * H * I
F_routed_moe_layer = K * 3 * 2 * B * S * H * E
F_shared_expert_layer = G * 3 * 2 * B * S * H * E_shared
```

The factor `3` is gate, up, and down projection. The factor `2` is multiply-add FLOPs.

Approximate weight bytes:

```text
B_dense_mlp_weights_layer = 3 * H * I * dtype_bytes
B_routed_expert_weights_layer = K * 3 * H * E * dtype_bytes
B_shared_expert_weights_layer = G * 3 * H * E_shared * dtype_bytes
```

Router cost is usually small but should be explicit:

```text
F_router_layer ~= 2 * B * S * H * num_total_experts
B_router_weights_layer ~= H * num_total_experts * dtype_bytes
```

If router logits are not materialized or the implementation fuses routing differently, keep this term optional and report whether it is included.

## Qwen3-Next Formula

Relevant config fields from the HF config:

```text
model_type = qwen3_next
num_hidden_layers = 48
hidden_size = 2048
intermediate_size = 5120
moe_intermediate_size = 512
num_experts = 512
num_experts_per_tok = 10
shared_expert_intermediate_size = 512
full_attention_interval = 4
linear_num_key_heads = 16
linear_num_value_heads = 32
linear_key_head_dim = 128
linear_value_head_dim = 128
num_attention_heads = 16
num_key_value_heads = 2
head_dim = 256
```

Qwen3-Next alternates mostly linear attention with periodic full attention. If layer indices are zero-based and every `full_attention_interval` layer is full attention, then:

```text
L_full = count(layer_id where (layer_id + 1) % full_attention_interval == 0)
L_linear = num_hidden_layers - L_full
```

For 48 layers and interval 4:

```text
L_full = 12
L_linear = 36
```

Total FLOPs:

```text
F_qwen3_next =
    L_full   * F_full_attention_layer(B, S, H, num_attention_heads, num_key_value_heads, head_dim)
  + L_linear * F_linear_attention_layer(B, S, H, linear_num_key_heads, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim)
  + 48 * F_routed_moe_layer(K = 10, E = 512)
  + 48 * F_shared_expert_layer(G = 1, E_shared = 512)
  + 48 * F_router_layer(num_total_experts = 512)
```

Suggested attention approximations:

```text
F_full_attention_layer =
    2 * B * S * H * (num_q_heads * head_dim)       # Q projection
  + 2 * B * S * H * (num_kv_heads * head_dim)      # K projection
  + 2 * B * S * H * (num_kv_heads * head_dim)      # V projection
  + 2 * B * S * S * num_q_heads * head_dim         # QK scores
  + 2 * B * S * S * num_q_heads * head_dim         # AV
  + 2 * B * S * (num_q_heads * head_dim) * H       # output projection
```

For linear attention, avoid using the `S^2` term:

```text
F_linear_attention_layer =
    F_linear_qkv_projections
  + F_linear_recurrent_or_state_update
  + F_linear_output_projection
  + F_short_conv
```

A practical first approximation is:

```text
F_linear_attention_layer ~=
    2 * B * S * H * ((linear_num_key_heads * linear_key_head_dim)
                   + (linear_num_value_heads * linear_value_head_dim)
                   + H)
  + 2 * B * S * H * H
```

This deliberately keeps linear attention O(B * S), not O(B * S^2).

Total bytes:

```text
B_qwen3_next =
    L_full   * B_full_attention_layer
  + L_linear * B_linear_attention_layer
  + 48 * B_routed_expert_weights_layer(K = 10, E = 512)
  + 48 * B_shared_expert_weights_layer(G = 1, E_shared = 512)
  + 48 * B_router_weights_layer(num_total_experts = 512)
  + B_kv_or_linear_state
```

Important implementation note: do not route Qwen3-Next through a plain Qwen3-MoE formula. Its MoE part is sparse, but its attention is hybrid.

## DeepSeek V4 Formula

DeepSeek V4 has Flash and Pro variants. The shared architecture signal is `model_type = deepseek_v4`.

DeepSeek-V4-Flash fields:

```text
num_hidden_layers = 43
hidden_size = 4096
moe_intermediate_size = 2048
n_routed_experts = 256
n_shared_experts = 1
num_experts_per_tok = 6
num_hash_layers = 3
sliding_window = 128
index_topk = 512
num_attention_heads = 64
num_key_value_heads = 1
head_dim = 512
compress_ratios = [0, 0, 4, 128, ... , 0]
```

DeepSeek-V4-Pro fields:

```text
num_hidden_layers = 61
hidden_size = 7168
moe_intermediate_size = 3072
n_routed_experts = 384
n_shared_experts = 1
num_experts_per_tok = 6
num_hash_layers = 3
sliding_window = 128
index_topk = 1024
num_attention_heads = 128
num_key_value_heads = 1
head_dim = 512
compress_ratios = [128, 128, 4, 128, ... , 0]
```

Layer split:

```text
L_hash_moe = num_hash_layers
L_routed_moe = num_hidden_layers - num_hash_layers
```

The hash MoE layers still activate a sparse subset of experts, but the selection is hash/table based rather than router-logit based. Count activated expert compute, not all experts:

```text
F_hash_moe_layer = K_hash * 3 * 2 * B * S * H * E
F_routed_moe_layer = K * 3 * 2 * B * S * H * E
F_shared_expert_layer = n_shared_experts * 3 * 2 * B * S * H * E_shared
```

Use `K = num_experts_per_tok = 6`. If hash layers use the same active count, set `K_hash = 6`. If the implementation exposes a different hash fanout, use that instead.

Total FLOPs:

```text
F_deepseek_v4 =
    sum_l F_deepseek_v4_attention_layer(l)
  + L_hash_moe   * F_hash_moe_layer(K_hash = 6)
  + L_routed_moe * F_routed_moe_layer(K = 6)
  + num_hidden_layers * F_shared_expert_layer(G = n_shared_experts)
  + L_routed_moe * F_router_layer(num_total_experts = n_routed_experts)
```

DeepSeek V4 attention should not be counted as ordinary dense full attention at 1M context. Use compressed/sliding sparse terms:

```text
F_deepseek_v4_attention_layer(l) =
    F_qkv_low_rank_or_projected
  + F_sliding_attention(window = sliding_window)
  + F_compressed_sparse_attention(compress_ratio = compress_ratios[l], index_topk = index_topk)
  + F_output_projection
```

Suggested approximations:

```text
F_sliding_attention =
    4 * B * S * sliding_window * num_attention_heads * head_dim
```

The `4` covers QK and AV, each using multiply-add FLOPs.

For compressed sparse attention:

```text
S_compressed_l = ceil(S / compress_ratio_l), if compress_ratio_l > 0

F_compressed_sparse_attention_l =
    4 * B * S * min(index_topk, S_compressed_l) * num_attention_heads * head_dim
```

If `compress_ratio_l == 0`, treat the compressed sparse term as disabled for that layer unless the implementation says otherwise.

Total bytes:

```text
B_deepseek_v4 =
    sum_l B_deepseek_v4_attention_layer(l)
  + L_hash_moe   * B_routed_expert_weights_layer(K_hash, E)
  + L_routed_moe * B_routed_expert_weights_layer(K, E)
  + num_hidden_layers * B_shared_expert_weights_layer(G, E_shared)
  + L_routed_moe * B_router_weights_layer(n_routed_experts)
  + B_sliding_cache
  + B_compressed_cache
```

For attention bytes:

```text
B_sliding_cache_layer ~= B * S * sliding_window * num_key_value_heads * head_dim * dtype_bytes
B_compressed_cache_layer ~= B * S_compressed_l * num_key_value_heads * head_dim * dtype_bytes
```

If the actual implementation stores compressed KV in a different latent dimension, replace `num_key_value_heads * head_dim` with that latent dimension.

## Reporting Rules

Report these fields with every result:

```text
model_type
num_layers
attention_layer_split
num_routed_experts
num_active_experts_per_token
num_shared_experts
moe_intermediate_size
sequence_length
batch_size
num_gpus
dtype_bytes
TTFT
F_total
B_total
TFLOPs_per_s
S_MFU
S_MBU
arithmetic_intensity
```

For comparisons, mark the formula as:

```text
exact: derived from model implementation and validated against MoE-CAP/server records
approximate: derived from HF config only
unsupported: missing architecture-specific attention or routing model
```

Until the attention kernels are matched to the real implementation, Qwen3-Next and DeepSeek V4 should be reported as `approximate`, not `exact`.
