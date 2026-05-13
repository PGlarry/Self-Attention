"""
s3_causal_probe.py — Module 3: Structured Approximation as Causal Probe
Usage:
    python s3_causal_probe.py <model_path> <model_tag> [task_id[,task_id...]] [layer head]
    python s3_causal_probe.py <model_path> <model_tag> [task_id[,task_id...]] <targets_csv>
    python s3_causal_probe.py <model_path> <model_tag> [task_id[,task_id...]] <targets_csv> --intervention plr --rank 2

Default task_id: TR01 (one representative prompt is enough for the probe)
Optionally specify a different task, e.g.: python s3_causal_probe.py ... TC10
Optionally specify multiple tasks, e.g.: python s3_causal_probe.py ... TR01,BZ01,TC01 targets.csv
Optionally scan one head only, e.g.: python s3_causal_probe.py ... TR01 0 0
Optionally scan selected heads from CSV, e.g.:
    python s3_causal_probe.py ... TR01 results/probe/probe_targets.csv

Method:
    For each (layer, head), replace the attention matrix mid-forward with a
    structured approximation (banded, w=5 — strongly local constraint by default), then
    measure how much the output token distribution shifts (KL divergence vs baseline).

    High KL → head depended heavily on global interactions → global integrator
    Low  KL → head is naturally local → local processor

This scan covers all heads in all layers for one prompt, producing a
"causal dependency map" of the model's attention heads.

Output: results/probe/probe_{model_tag}_{task_id}.csv
"""
import sys
import csv
import gc
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

if len(sys.argv) < 3:
    print("Usage: python s3_causal_probe.py <model_path> <model_tag> [task_id] [layer head] [--intervention banded|plr] [--rank 2]")
    sys.exit(1)

raw_args = sys.argv[1:]
MODEL_PATH = raw_args[0]
MODEL_TAG  = raw_args[1]
remaining_args = raw_args[2:]
positional_args = []
INTERVENTION_TYPE = "banded"
PLR_RANK = 2
PROBE_BAND_WIDTH = 5   # half-width for the locality constraint applied as intervention

i = 0
while i < len(remaining_args):
    arg = remaining_args[i]
    if arg == "--intervention":
        INTERVENTION_TYPE = remaining_args[i + 1].strip().lower()
        i += 2
    elif arg == "--rank":
        PLR_RANK = int(remaining_args[i + 1])
        i += 2
    elif arg == "--band-width":
        PROBE_BAND_WIDTH = int(remaining_args[i + 1])
        i += 2
    else:
        positional_args.append(arg)
        i += 1

if INTERVENTION_TYPE not in {"banded", "plr"}:
    print(f"Unsupported intervention: {INTERVENTION_TYPE}")
    sys.exit(1)

TASK_ID    = positional_args[0] if len(positional_args) > 0 else "TR01"
TASK_IDS   = [item.strip() for item in TASK_ID.split(",") if item.strip()]
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_CSV = None
TARGET_LAYER = None
TARGET_HEAD = None
if len(positional_args) > 1:
    if len(positional_args) == 2 and positional_args[1].lower().endswith(".csv"):
        TARGET_CSV = Path(positional_args[1])
    else:
        TARGET_LAYER = int(positional_args[1])
        TARGET_HEAD = int(positional_args[2]) if len(positional_args) > 2 else None

PROMPT_DIR = PROJECT_ROOT / "prompts"
TASK_TAG   = "multi" if len(TASK_IDS) > 1 else TASK_IDS[0]
INTERVENTION_SUFFIX = "" if INTERVENTION_TYPE == "banded" else f"_{INTERVENTION_TYPE}_r{PLR_RANK}"
OUT_CSV    = PROJECT_ROOT / "results" / "probe" / f"probe_{MODEL_TAG}_{TASK_TAG}{INTERVENTION_SUFFIX}.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

ACTIVE_INTERVENTION = {
    "layer": None,
    "head": None,
    "intervention_type": INTERVENTION_TYPE,
    "band_width": PROBE_BAND_WIDTH,
    "plr_rank": PLR_RANK,
    "fired": False,
}


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match query heads, following HF Llama-style attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def approx_banded(A: torch.Tensor, half_width: int) -> torch.Tensor:
    N = A.shape[0]
    rows = torch.arange(N, device=A.device).unsqueeze(1)
    cols = torch.arange(N, device=A.device).unsqueeze(0)
    mask = ((rows - cols).abs() <= half_width).float()
    A_approx = A.float() * mask
    row_sums = A_approx.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return (A_approx / row_sums).to(A.dtype)


def approx_banded_batch(A: torch.Tensor, half_width: int) -> torch.Tensor:
    """Banded approximation for a batch of attention matrices: (batch, seq, seq)."""
    N = A.shape[-1]
    rows = torch.arange(N, device=A.device).unsqueeze(1)
    cols = torch.arange(N, device=A.device).unsqueeze(0)
    mask = ((rows - cols).abs() <= half_width).to(A.dtype)
    A_approx = A * mask
    row_sums = A_approx.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return A_approx / row_sums


def approx_plr_batch(A: torch.Tensor, rank: int) -> torch.Tensor:
    """Projected low-rank approximation for attention matrices: (batch, seq, seq)."""
    approx_rows = []
    for item in A:
        U, S, Vh = torch.linalg.svd(item.float(), full_matrices=False)
        r = min(rank, S.shape[0])
        A_approx = (U[:, :r] * S[:r]) @ Vh[:r, :]
        A_approx = A_approx.clamp(min=0.0)
        row_sums = A_approx.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        approx_rows.append((A_approx / row_sums).to(A.dtype))
    return torch.stack(approx_rows, dim=0)


def probe_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Eager attention with an optional one-head structured intervention.

    The replacement happens before attn_output = A @ V, so the intervention
    changes hidden states and final logits, not just the returned diagnostics.
    """
    if scaling is None:
        scaling = module.head_dim ** -0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if softcap is not None:
        attn_weights = torch.tanh(attn_weights / softcap) * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    target_layer = ACTIVE_INTERVENTION["layer"]
    target_head = ACTIVE_INTERVENTION["head"]
    if target_layer is not None and getattr(module, "layer_idx", None) == target_layer:
        if target_head is not None and target_head < attn_weights.shape[1]:
            attn_weights = attn_weights.clone()
            if ACTIVE_INTERVENTION["intervention_type"] == "plr":
                attn_weights[:, target_head, :, :] = approx_plr_batch(
                    attn_weights[:, target_head, :, :],
                    int(ACTIVE_INTERVENTION["plr_rank"]),
                )
            else:
                attn_weights[:, target_head, :, :] = approx_banded_batch(
                    attn_weights[:, target_head, :, :],
                    int(ACTIVE_INTERVENTION["band_width"]),
                )
            ACTIVE_INTERVENTION["fired"] = True

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def kl_divergence(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-12) -> float:
    """KL(P || Q) for two 1-D probability vectors."""
    P = P.float().clamp(min=eps)
    Q = Q.float().clamp(min=eps)
    P = P / P.sum()
    Q = Q / Q.sum()
    return (P * (P / Q).log()).sum().item()


def js_divergence(P: torch.Tensor, Q: torch.Tensor) -> float:
    """Jensen-Shannon divergence (symmetric, bounded in [0, log2])."""
    P = P.float().clamp(min=1e-12)
    Q = Q.float().clamp(min=1e-12)
    P = P / P.sum()
    Q = Q / Q.sum()
    M = 0.5 * (P + Q)
    return 0.5 * (P * (P / M).log()).sum().item() + \
           0.5 * (Q * (Q / M).log()).sum().item()


def top_k_overlap(P: torch.Tensor, Q: torch.Tensor, k: int = 10) -> float:
    """Fraction of top-k tokens in P that also appear in top-k of Q."""
    topk_P = set(P.topk(k).indices.tolist())
    topk_Q = set(Q.topk(k).indices.tolist())
    return len(topk_P & topk_Q) / k


def load_tokenizer(model_path: str):
    kwargs = {}
    if "mistral" in model_path.lower() or "ministral" in model_path.lower():
        kwargs["fix_mistral_regex"] = True
    return AutoTokenizer.from_pretrained(model_path, **kwargs)


def load_target_rows(path: Path) -> list[dict]:
    targets = []
    with open(path, "r", newline="", encoding="utf-8-sig") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            if row.get("model_tag") != MODEL_TAG:
                continue
            targets.append({
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "target_group": row.get("target_group", ""),
                "selection_rank": row.get("selection_rank", ""),
                "source_erank": row.get("erank", ""),
                "source_diag_conc": row.get("diag_conc", ""),
                "source_mean_dist": row.get("mean_dist", ""),
            })
    return targets


def main():
    print(f"Model   : {MODEL_PATH}")
    print(f"Tag     : {MODEL_TAG}")
    print(f"Tasks   : {','.join(TASK_IDS)}")
    print(f"Interv. : {INTERVENTION_TYPE}" + (f" rank={PLR_RANK}" if INTERVENTION_TYPE == "plr" else f" band_width={PROBE_BAND_WIDTH}"))
    print(f"Output  : {OUT_CSV}")
    if TARGET_CSV is not None:
        print(f"Targets : {TARGET_CSV}")

    prompt_texts = {}
    for task_id in TASK_IDS:
        prompt_path = PROMPT_DIR / f"{task_id}__FULL.txt"
        if not prompt_path.exists():
            print(f"Prompt file not found: {prompt_path}")
            sys.exit(1)
        prompt_texts[task_id] = prompt_path.read_text(encoding="utf-8")

    tok = load_tokenizer(MODEL_PATH)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    ALL_ATTENTION_FUNCTIONS.register("probe_eager", probe_attention_forward)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="probe_eager",
    )
    model.eval()
    if torch.cuda.is_available():
        print(f"VRAM    : {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # ── Function-level intervention scan ──────────────────────────────────────
    fields = [
        "task_id", "model_tag", "layer", "head", "seq_len",
        "intervention_type", "band_width", "plr_rank",
        "target_group", "selection_rank",
        "source_erank", "source_diag_conc", "source_mean_dist",
        "intervention_fired", "kl_div", "js_div", "top10_overlap",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()

        for task_id, prompt_text in prompt_texts.items():
            inputs  = tok(prompt_text, return_tensors="pt").to(DEVICE)
            seq_len = inputs["input_ids"].shape[1]
            print(f"\n[{task_id}] seq_len={seq_len}")

            print("Running baseline forward pass ...", flush=True)
            with torch.no_grad():
                base_out = model(**inputs, output_attentions=True)

            base_logits = base_out.logits[0, -1, :].float()
            base_probs  = F.softmax(base_logits, dim=-1).cpu()
            n_layers    = len(base_out.attentions)
            n_heads     = base_out.attentions[0].shape[1]
            print(f"n_layers={n_layers}  n_heads={n_heads}")

            target_rows = None
            if TARGET_CSV is not None:
                if not TARGET_CSV.exists():
                    print(f"Target CSV not found: {TARGET_CSV}")
                    sys.exit(1)
                target_rows = load_target_rows(TARGET_CSV)
                if not target_rows:
                    print(f"No target rows for model_tag={MODEL_TAG} in {TARGET_CSV}")
                    sys.exit(1)
                for row in target_rows:
                    if row["layer"] < 0 or row["layer"] >= n_layers or row["head"] < 0 or row["head"] >= n_heads:
                        print(f"Invalid target row: layer={row['layer']}, head={row['head']}")
                        sys.exit(1)
                print(f"selected-head scan: {len(target_rows)} heads")
            elif TARGET_LAYER is not None or TARGET_HEAD is not None:
                if TARGET_LAYER is None or TARGET_HEAD is None:
                    print("Both layer and head must be provided for a single-head scan.")
                    sys.exit(1)
                if TARGET_LAYER < 0 or TARGET_LAYER >= n_layers or TARGET_HEAD < 0 or TARGET_HEAD >= n_heads:
                    print(f"Invalid target layer/head: layer={TARGET_LAYER}, head={TARGET_HEAD}")
                    sys.exit(1)
                print(f"single-head scan: layer={TARGET_LAYER}, head={TARGET_HEAD}")

            del base_out

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if target_rows is not None:
                scan_rows = target_rows
            elif TARGET_LAYER is not None:
                scan_rows = [{"layer": TARGET_LAYER, "head": TARGET_HEAD}]
            else:
                scan_rows = [
                    {"layer": layer_idx, "head": head_idx}
                    for layer_idx in range(n_layers)
                    for head_idx in range(n_heads)
                ]

            current_layer = None
            for scan_row in scan_rows:
                layer_idx = int(scan_row["layer"])
                head_idx = int(scan_row["head"])
                if layer_idx != current_layer:
                    if current_layer is not None:
                        print("done")
                    current_layer = layer_idx
                    print(f"  Layer {layer_idx:03d}/{n_layers} ...", end=" ", flush=True)

                target_group = scan_row.get("target_group", "")
                selection_rank = scan_row.get("selection_rank", "")
                source_erank = scan_row.get("source_erank", "")
                source_diag_conc = scan_row.get("source_diag_conc", "")
                source_mean_dist = scan_row.get("source_mean_dist", "")

                ACTIVE_INTERVENTION.update({
                    "layer": layer_idx,
                    "head": head_idx,
                    "intervention_type": INTERVENTION_TYPE,
                    "band_width": PROBE_BAND_WIDTH,
                    "plr_rank": PLR_RANK,
                    "fired": False,
                })

                with torch.no_grad():
                    try:
                        intv_out = model(**inputs, output_attentions=True)
                        intv_logits = intv_out.logits[0, -1, :].float()
                        intv_probs  = F.softmax(intv_logits, dim=-1).cpu()
                        del intv_out
                    except Exception:
                        intv_probs = base_probs.clone()

                intervention_fired = bool(ACTIVE_INTERVENTION["fired"])
                ACTIVE_INTERVENTION.update({"layer": None, "head": None, "fired": False})

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                kl   = kl_divergence(base_probs, intv_probs)
                js   = js_divergence(base_probs, intv_probs)
                ov10 = top_k_overlap(base_probs, intv_probs, k=10)

                writer.writerow({
                    "task_id":     task_id,
                    "model_tag":   MODEL_TAG,
                    "layer":       layer_idx,
                    "head":        head_idx,
                    "seq_len":     seq_len,
                    "intervention_type": INTERVENTION_TYPE,
                    "band_width":  PROBE_BAND_WIDTH,
                    "plr_rank":    PLR_RANK if INTERVENTION_TYPE == "plr" else "",
                    "target_group": target_group,
                    "selection_rank": selection_rank,
                    "source_erank": source_erank,
                    "source_diag_conc": source_diag_conc,
                    "source_mean_dist": source_mean_dist,
                    "intervention_fired": int(intervention_fired),
                    "kl_div":      round(kl, 6),
                    "js_div":      round(js, 6),
                    "top10_overlap": round(ov10, 4),
                })
                fout.flush()

            if current_layer is not None:
                print("done")

    print(f"\n✅ Done → {OUT_CSV}")


if __name__ == "__main__":
    main()
