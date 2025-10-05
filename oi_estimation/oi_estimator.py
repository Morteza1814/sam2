"""
Operational Intensity (OI) calculator for SAM2 MemoryAttention.

Assumptions (as requested):
- Each step reads its inputs from memory, computes, and WRITES the outputs back.
- Activation is ReLU.
- Ignore object-pointer tokens for now (num_obj_ptr_tokens = 0).
- Defaults config: H=1, d_model=256, d_ff=2048, kv_in_dim=64, num_layers=4.
- Lq = 4096; Lk = num_maskmem * 4096 (default num_maskmem=7).
- B (batch size) default = 1; dtype_bytes default = 2 (fp16/bf16).

Notation:
- FLOPs count uses multiply+add = 2 FLOPs for GEMMs.
- Softmax ~ 5 FLOPs per element (stable-softmax rule of thumb).
- LayerNorm ~ 5 FLOPs per element.
- ReLU ~ 1 FLOP per element.
- RoPE ~ 3 FLOPs per scalar (rotation on Q and K).
"""

from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Tuple
from sys import argv
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class MAParams:
    # Model/attention dimensions
    B: int = 1
    Lq: int = 4096                   # current tokens length
    Lk: int = 7 * 4096               # memory tokens length (num_maskmem * 4096)
    d_model: int = 256
    d_ff: int = 2048
    H: int = 1                       # number of heads
    kv_in_dim: int = 64              # memory_encoder.out_dim; K/V input channels for cross-attn
    num_layers: int = 4

    # Data type
    dtype_bytes: int = 2             # 2 for fp16/bf16, 4 for fp32

    # Ops toggles (match your config)
    activation: str = "relu"         # ReLU FLOPs ~ 1/elt
    use_rope_sa_q: bool = True       # apply RoPE to Q in self-attn
    use_rope_sa_k: bool = True       # apply RoPE to K in self-attn
    use_rope_ca_q: bool = False      # per config pos_enc_at_cross_attn_queries: false
    use_rope_ca_k: bool = True       # per config pos_enc_at_cross_attn_keys: true
    include_rope_tables_read: bool = False  # if True, count reading sin/cos tables (same size as Q/K)

    # MemoryAttention wrapper flags (from your config)
    pos_enc_at_input: bool = True    # adds curr_pos to input once at the top
    include_final_norm: bool = True  # MemoryAttention.norm at the end


# -------------------- FLOP helpers --------------------

def flops_linear(N: int, d_in: int, d_out: int) -> int:
    # GEMM multiply+add
    return 2 * N * d_in * d_out

def flops_relu(N: int) -> int:
    return N  # ~1 FLOP per element

def flops_layernorm(N: int) -> int:
    return 5 * N  # mean + var + normalize + scale + shift (rule of thumb)

def flops_softmax(n_rows: int, row_len: int) -> int:
    return 5 * n_rows * row_len  # log-sum-exp-ish rule of thumb

def flops_rope(n_elts: int) -> int:
    return 3 * n_elts  # ~3 FLOPs per scalar


# -------------------- BYTES helpers --------------------

def bytes_linear_activations(dtype_bytes: int, N: int, d_in: int, d_out: int) -> int:
    # Read input, write output (we count the write once here;
    # reading/writing of intermediate buffers is handled step-by-step by the main flow)
    return dtype_bytes * (N * d_in + N * d_out)

def bytes_linear_params(dtype_bytes: int, d_in: int, d_out: int, bias: bool = True) -> int:
    return dtype_bytes * (d_in * d_out + (d_out if bias else 0))

def bytes_elementwise(dtype_bytes: int, read_elts: int, write_elts: int) -> int:
    return dtype_bytes * (read_elts + write_elts)

def bytes_layernorm_params(dtype_bytes: int, d_model: int) -> int:
    # gamma + beta
    return dtype_bytes * (2 * d_model)

def bytes_softmax(dtype_bytes: int, n_scores: int) -> int:
    # Read scores, write prob
    return dtype_bytes * (n_scores + n_scores)

def bytes_rope(dtype_bytes: int, n_elts: int, include_tables: bool) -> int:
    # Read input + write output (+ optional sin/cos tables)
    return dtype_bytes * (n_elts + n_elts + (n_elts if include_tables else 0))


# -------------------- Attention block accounting --------------------

def account_self_attention(params: MAParams) -> Dict[str, Dict[str, int]]:
    """
    Self-attention on sequence of length Lq with H heads (H=1 by default).
    We treat each step as: read inputs + read weights -> compute -> write outputs.
    Then the next step re-reads the outputs it needs.
    """
    B, Lq, H = params.B, params.Lq, params.H
    dm = params.d_model
    dk = dm // H  # head dim
    s = params.dtype_bytes

    report: Dict[str, Dict[str, int]] = {}

    Nq = B * Lq  # tokens count

    # 1) LayerNorm on tgt (pre-norm)
    step = "sa_ln"
    flops = flops_layernorm(Nq * dm)
    bytes_rw = bytes_elementwise(s, read_elts=Nq * dm, write_elts=Nq * dm) \
               + bytes_layernorm_params(s, dm)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 2) Linear Q (d_model -> d_model) on normalized tgt
    step = "sa_q_proj"
    flops = flops_linear(Nq, dm, dm)
    bytes_rw = bytes_linear_activations(s, Nq, dm, dm) + bytes_linear_params(s, dm, dm, True)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 3) Linear K
    step = "sa_k_proj"
    flops = flops_linear(Nq, dm, dm)
    bytes_rw = bytes_linear_activations(s, Nq, dm, dm) + bytes_linear_params(s, dm, dm, True)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 4) Linear V
    step = "sa_v_proj"
    flops = flops_linear(Nq, dm, dm)
    bytes_rw = bytes_linear_activations(s, Nq, dm, dm) + bytes_linear_params(s, dm, dm, True)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 5) RoPE on Q (optional)
    if params.use_rope_sa_q:
        step = "sa_rope_q"
        n_elts = B * H * Lq * dk
        flops = flops_rope(n_elts)
        bytes_rw = bytes_rope(s, n_elts, params.include_rope_tables_read)
        report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 6) RoPE on K (optional)
    if params.use_rope_sa_k:
        step = "sa_rope_k"
        n_elts = B * H * Lq * dk
        flops = flops_rope(n_elts)
        bytes_rw = bytes_rope(s, n_elts, params.include_rope_tables_read)
        report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 7) Scores = Q @ K^T  (Lq x Lq per head)
    step = "sa_qk_matmul_write_scores"
    flops = 2 * B * H * Lq * Lq * dk
    n_scores = B * H * Lq * Lq
    # Read Q and K, write Scores
    bytes_rw = s * (B * H * Lq * dk + B * H * Lq * dk + n_scores)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 8) Read scores and softmax them (we assume scores were written and now re-read)
    step = "sa_softmax"
    flops = flops_softmax(B * H * Lq, Lq)
    bytes_rw = bytes_softmax(s, n_scores)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 9) Context = Prob @ V  (Lq x dk)
    step = "sa_av_matmul"
    flops = 2 * B * H * Lq * Lq * dk
    # Read Prob and V, write Context
    bytes_rw = s * (n_scores + B * H * Lq * dk + B * H * Lq * dk)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 10) Output projection (concat heads already; H=1)
    step = "sa_out_proj"
    # Context shape [B, Lq, dm]
    flops = flops_linear(Nq, dm, dm)
    bytes_rw = bytes_linear_activations(s, Nq, dm, dm) + bytes_linear_params(s, dm, dm, True)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 11) Residual add: tgt <- tgt + sa_out
    step = "sa_residual_add"
    flops = Nq * dm
    bytes_rw = s * (Nq * dm + Nq * dm + Nq * dm)  # read two, write one
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    return report


def account_cross_attention(params: MAParams) -> Dict[str, Dict[str, int]]:
    """
    Cross-attention: Q from current (Lq), K/V from memory (Lk) with kv_in_dim->d_model projection.
    """
    B, Lq, Lk, H = params.B, params.Lq, params.Lk, params.H
    dm, kvd = params.d_model, params.kv_in_dim
    dk = dm // H
    s = params.dtype_bytes

    report: Dict[str, Dict[str, int]] = {}

    Nq = B * Lq
    Nk = B * Lk

    # 1) LayerNorm on tgt (pre-norm)
    step = "ca_ln"
    flops = flops_layernorm(Nq * dm)
    bytes_rw = bytes_elementwise(s, read_elts=Nq * dm, write_elts=Nq * dm) \
               + bytes_layernorm_params(s, dm)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 2) Q projection (d_model -> d_model) on tgt2
    step = "ca_q_proj"
    flops = flops_linear(Nq, dm, dm)
    bytes_rw = bytes_linear_activations(s, Nq, dm, dm) + bytes_linear_params(s, dm, dm, True)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 3) K projection (kv_in_dim -> d_model) on memory
    step = "ca_k_proj"
    flops = flops_linear(Nk, kvd, dm)
    bytes_rw = bytes_linear_activations(s, Nk, kvd, dm) + bytes_linear_params(s, kvd, dm, True)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 4) V projection (kv_in_dim -> d_model) on memory
    step = "ca_v_proj"
    flops = flops_linear(Nk, kvd, dm)
    bytes_rw = bytes_linear_activations(s, Nk, kvd, dm) + bytes_linear_params(s, kvd, dm, True)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 5) RoPE on Q (optional per config: default False)
    if params.use_rope_ca_q:
        step = "ca_rope_q"
        n_elts = B * H * Lq * dk
        flops = flops_rope(n_elts)
        bytes_rw = bytes_rope(s, n_elts, params.include_rope_tables_read)
        report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 6) RoPE on K (optional per config: default True)
    if params.use_rope_ca_k:
        step = "ca_rope_k"
        n_elts = B * H * Lk * dk
        flops = flops_rope(n_elts)
        bytes_rw = bytes_rope(s, n_elts, params.include_rope_tables_read)
        report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 7) Scores = Q @ K^T  (Lq x Lk per head)
    step = "ca_qk_matmul_write_scores"
    flops = 2 * B * H * Lq * Lk * dk
    n_scores = B * H * Lq * Lk
    bytes_rw = s * (B * H * Lq * dk + B * H * Lk * dk + n_scores)  # read Q,K; write Scores
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 8) Read scores and softmax
    step = "ca_softmax"
    flops = flops_softmax(B * H * Lq, Lk)
    bytes_rw = bytes_softmax(s, n_scores)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 9) Context = Prob @ V  (Lq x dk)
    step = "ca_av_matmul"
    flops = 2 * B * H * Lq * Lk * dk
    bytes_rw = s * (n_scores + B * H * Lk * dk + B * H * Lq * dk)  # read Prob, V; write Context
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 10) Output projection
    step = "ca_out_proj"
    flops = flops_linear(Nq, dm, dm)
    bytes_rw = bytes_linear_activations(s, Nq, dm, dm) + bytes_linear_params(s, dm, dm, True)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 11) Residual add: tgt <- tgt + ca_out
    step = "ca_residual_add"
    flops = Nq * dm
    bytes_rw = s * (Nq * dm + Nq * dm + Nq * dm)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    return report


def account_mlp(params: MAParams) -> Dict[str, Dict[str, int]]:
    """
    MLP: LN -> Linear(d_model->d_ff) -> ReLU -> Linear(d_ff->d_model) -> Residual add
    """
    B, Lq = params.B, params.Lq
    dm, dff = params.d_model, params.d_ff
    s = params.dtype_bytes

    report: Dict[str, Dict[str, int]] = {}
    N = B * Lq

    # 1) LayerNorm
    step = "mlp_ln"
    flops = flops_layernorm(N * dm)
    bytes_rw = bytes_elementwise(s, read_elts=N * dm, write_elts=N * dm) \
               + bytes_layernorm_params(s, dm)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 2) Linear1: d_model -> d_ff
    step = "mlp_linear1"
    flops = flops_linear(N, dm, dff)
    bytes_rw = bytes_linear_activations(s, N, dm, dff) + bytes_linear_params(s, dm, dff, True)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 3) ReLU activation
    step = "mlp_relu"
    flops = flops_relu(N * dff) if params.activation.lower() == "relu" else flops_relu(N * dff)
    bytes_rw = bytes_elementwise(s, read_elts=N * dff, write_elts=N * dff)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 4) Linear2: d_ff -> d_model
    step = "mlp_linear2"
    flops = flops_linear(N, dff, dm)
    bytes_rw = bytes_linear_activations(s, N, dff, dm) + bytes_linear_params(s, dff, dm, True)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    # 5) Residual add
    step = "mlp_residual_add"
    flops = N * dm
    bytes_rw = s * (N * dm + N * dm + N * dm)
    report[step] = {"FLOPs": flops, "Bytes": bytes_rw}

    return report


def account_posenc_at_input(params: MAParams) -> Dict[str, Dict[str, int]]:
    """
    If pos_enc_at_input is True, MemoryAttention adds 0.1 * curr_pos to the input once.
    We count this as an elementwise add (scale + add, but we conservatively count as 1 add).
    """
    if not params.pos_enc_at_input:
        return {}
    B, Lq, dm, s = params.B, params.Lq, params.d_model, params.dtype_bytes
    N = B * Lq
    flops = N * dm
    bytes_rw = s * (N * dm + N * dm + N * dm)  # read curr + pos, write out
    return {"input_posenc_add": {"FLOPs": flops, "Bytes": bytes_rw}}


def account_final_norm(params: MAParams) -> Dict[str, Dict[str, int]]:
    if not params.include_final_norm:
        return {}
    B, Lq, dm, s = params.B, params.Lq, params.d_model, params.dtype_bytes
    N = B * Lq
    flops = flops_layernorm(N * dm)
    bytes_rw = bytes_elementwise(s, read_elts=N * dm, write_elts=N * dm) \
               + bytes_layernorm_params(s, dm)
    return {"final_norm": {"FLOPs": flops, "Bytes": bytes_rw}}


# -------------------- Aggregation utilities --------------------

def add_sections(sec_a: Dict[str, Dict[str, int]], sec_b: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    out = dict(sec_a)
    for k, v in sec_b.items():
        out[k] = {"FLOPs": v["FLOPs"], "Bytes": v["Bytes"]}
    return out

def totals(section: Dict[str, Dict[str, int]]) -> Tuple[int, int]:
    f = sum(v["FLOPs"] for v in section.values())
    b = sum(v["Bytes"] for v in section.values())
    return f, b

def oi(f: int, b: int) -> float:
    return (f / b) if b > 0 else float("inf")


# -------------------- Public API --------------------

def memory_attention_layer_profile(params: MAParams) -> Dict:
    """
    One MemoryAttentionLayer: self-attn + cross-attn + MLP, each with LN and residual per the code.
    """
    sections = {}
    sections.update(account_posenc_at_input(params))  # only counted once at the very start (outside layer loop)

    sa = account_self_attention(params)
    ca = account_cross_attention(params)
    mlp = account_mlp(params)

    layer = {}
    layer.update(sa)
    layer.update(ca)
    layer.update(mlp)

    f_layer, b_layer = totals(layer)
    layer["__TOTAL__"] = {"FLOPs": f_layer, "Bytes": b_layer, "OI": oi(f_layer, b_layer)}

    out = {"per_layer": layer}
    return out


def memory_attention_stack_profile(params: MAParams) -> Dict:
    """
    Full MemoryAttention with num_layers identical layers + optional final norm.
    Counts the input pos-enc add ONCE at the very beginning.
    """

    # Input pos-enc add (once)
    pre = account_posenc_at_input(params)

    # Build a copy where we disable counting pos-enc inside each layer
    inner_params = replace(params, pos_enc_at_input=False)

    # Profile one layer with pos-enc disabled (we already counted it once above)
    per_layer = memory_attention_layer_profile(inner_params)["per_layer"]
    
    # Sum over num_layers
    f_layer, b_layer = totals(per_layer)
    f_all_layers = params.num_layers * f_layer
    b_all_layers = params.num_layers * b_layer

    # Final norm (once)
    post = account_final_norm(params)
    f_pre, b_pre = totals(pre)
    f_post, b_post = totals(post)

    f_total = f_pre + f_all_layers + f_post
    b_total = b_pre + b_all_layers + b_post

    report = {
        "params": asdict(params),
        "pre_once": pre,
        "per_layer": per_layer,
        "final_norm_once": post,
        "totals": {
            "FLOPs": f_total,
            "Bytes": b_total,
            "OI": oi(f_total, b_total),
            "FLOPs_per_layer": f_layer,
            "Bytes_per_layer": b_layer,
            "OI_per_layer": oi(f_layer, b_layer),
        },
    }
    return report

def _sum_flops_bytes(items):
    f = sum(v["FLOPs"] for v in items)
    b = sum(v["Bytes"] for v in items)
    return f, b

def _ridge(peak_tflops, peak_tbps):
    return peak_tflops / peak_tbps  # FLOPs/byte

def _marker_for(label):
    if label.startswith("sa_"): return '^'
    if label.startswith("ca_"): return 's'
    if label.startswith("mlp_"): return 'D'
    return 'o'

def _roofline_plot(peak_tflops, peak_tbps, label_to_FB, title):
    # Build roofline curves
    x = np.logspace(-2, 4, 400)               # FLOPs/byte
    y_mem = peak_tbps * x                     # TFLOP/s
    y_cmp = np.full_like(x, peak_tflops)      # TFLOP/s
    y = np.minimum(y_mem, y_cmp)
    ridge = _ridge(peak_tflops, peak_tbps)

    # Plot
    plt.figure(figsize=(8, 5.2))
    plt.loglog(x, y_mem, linestyle='--', linewidth=1.5, label='Bandwidth ceiling')
    plt.loglog(x, y_cmp, linestyle='--', linewidth=1.5, label='Compute ceiling')
    plt.loglog(x, y, linewidth=2.2, label='Roofline')
    plt.scatter([ridge], [peak_tflops], marker='x', s=60, label='Ridge')

    # Points
    printed = []
    for lab, (F, B) in label_to_FB.items():
        if B <= 0: 
            continue
        oi = F / B
        y_att = min(peak_tflops, peak_tbps * oi)
        marker = _marker_for(lab)
        plt.scatter([oi], [y_att], s=70, marker=marker, edgecolors='black', linewidths=0.8, label=lab)
        plt.axvline(oi, linestyle=':', linewidth=1.0)
        printed.append((lab, oi, y_att))

    plt.xlabel("Operational Intensity (FLOPs/byte)")
    plt.ylabel("Attainable Performance (TFLOP/s)")
    plt.title(title)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout()
    plt.show()
    plt.savefig("roofline_plot.png", dpi=300, bbox_inches='tight')

    # Print the numbers separately (keeps figure clean)
    print("=== Roofline context ===")
    print(f"Peaks: {peak_tflops:.0f} TFLOP/s, {peak_tbps:.2f} TB/s  |  ridge ≈ {ridge:.1f} FLOPs/byte")
    for lab, oi, y in printed:
        bound = "compute-bound" if oi >= ridge else "memory-bound"
        print(f"- {lab:24s}  OI = {oi:8.2f}  attainable ≈ {y:8.1f} TFLOP/s  → {bound}")

def _points_from_per_layer(per_layer: dict):
    """Return two dicts: (group_points, subkernel_points), each mapping name -> (FLOPs, Bytes)."""
    # Subkernels (everything except __TOTAL__)
    sub = {k: (v["FLOPs"], v["Bytes"]) for k, v in per_layer.items() if k != "__TOTAL__"}

    # Groups
    def pick(prefix):
        items = [(k, v) for k, v in per_layer.items() if k.startswith(prefix)]
        return _sum_flops_bytes(v for _, v in items)

    groups = {}
    if "__TOTAL__" in per_layer:
        groups["Module (per-layer total)"] = (per_layer["__TOTAL__"]["FLOPs"], per_layer["__TOTAL__"]["Bytes"])
    groups["Self-Attention"]  = pick("sa_")
    groups["Cross-Attention"] = pick("ca_")
    groups["MLP"]             = pick("mlp_")
    return groups, sub

def build_points_for_plots(params: MAParams, use_per_layer=True, include_final_norm_once=True):
    """
    Collect points for plots.
    - use_per_layer=True → profile ONE layer (no input pos-enc); good for 'module' view.
    - include_final_norm_once: if False, drop the final MemoryAttention.norm point (when plotting stack totals).
    Returns (group_points, subkernel_points).
    """
    if use_per_layer:
        # We already counted input pos-enc outside the layer in the stack profile;
        # for clean per-layer view, disable it here.
        inner_params = replace(params, pos_enc_at_input=False)
        per_layer = memory_attention_layer_profile(inner_params)["per_layer"]
    else:
        # Stack totals (num_layers), expanded to show a synthetic per-layer subdivision is not trivial,
        # so we still use per-layer breakdown for subkernels and just scale their FLOPs/Bytes.
        inner_params = replace(params, pos_enc_at_input=False)
        per_layer = memory_attention_layer_profile(inner_params)["per_layer"]
        # Scale by num_layers
        for k in per_layer:
            per_layer[k]["FLOPs"] *= params.num_layers
            per_layer[k]["Bytes"] *= params.num_layers
        if include_final_norm_once:
            # Add a 'final_norm' once
            extra = account_final_norm(params)
            per_layer.update(extra)

    groups, sub = _points_from_per_layer(per_layer)
    return groups, sub

def plot_memory_attention_rooflines(params: MAParams,
                                    peak_tflops=1979.0, peak_tbps=3.35,
                                    plot_groups=True, plot_subkernels=True,
                                    title_prefix="H100 SXM (FP16 Tensor Cores)"):
    """
    Draw two charts:
      1) Grouped points (Module total, Self-Attn, Cross-Attn, MLP)
      2) Sub-kernels (LN, Q/K/V proj, softmax, AV matmul, out-proj, ReLU, residual, etc.)
    """
    groups, sub = build_points_for_plots(params, use_per_layer=True)

    if plot_groups:
        _roofline_plot(
            peak_tflops, peak_tbps, groups,
            title=f"Roofline — {title_prefix}\nMemoryAttention (per-layer groups)"
        )

    if plot_subkernels:
        _roofline_plot(
            peak_tflops, peak_tbps, sub,
            title=f"Roofline — {title_prefix}\nMemoryAttention (per-layer sub-kernels)"
        )


if __name__ == "__main__":
    num_maskmem = int(argv[1]) if len(argv) > 1 else 7
    # Defaults aligned with your config and choices
    p = MAParams(
        B=1,
        Lq=4096,
        Lk=num_maskmem * 4096,   # ← replaces Lk=7 * 4096
        d_model=256,
        d_ff=2048,
        H=1,
        kv_in_dim=64,
        num_layers=4,
        dtype_bytes=2,     # fp16/bf16
        activation="relu",
        use_rope_sa_q=True,
        use_rope_sa_k=True,
        use_rope_ca_q=False,  # per your config
        use_rope_ca_k=True,   # per your config
        include_rope_tables_read=False,
        pos_enc_at_input=True,
        include_final_norm=True,
    )

    report = memory_attention_stack_profile(p)

    # Pretty print a compact summary
    print("=== MemoryAttention OI Report ===")
    print("Params:", report["params"])
    print("\n-- Per-layer totals --")
    print({k: report["totals"][k] for k in ("FLOPs_per_layer", "Bytes_per_layer", "OI_per_layer")})
    print("\n-- Full stack totals --")
    print({k: report["totals"][k] for k in ("FLOPs", "Bytes", "OI")})


    # H100 SXM peaks (FP16 Tensor Cores)
    plot_memory_attention_rooflines(
        p,
        peak_tflops=1979.0,
        peak_tbps=3.35,
        plot_groups=True,
        plot_subkernels=True,
        title_prefix="H100 SXM (FP16 Tensor Cores)"
    )


    # If you want to see detailed steps for one layer:
    # for name, vals in report["per_layer"].items():
    #     print(name, vals)

