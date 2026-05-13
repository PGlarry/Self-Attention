"""
s1_extract_spectra.py — Module 1: Spectral Analysis of Attention Matrices
Usage:
    python s1_extract_spectra.py <model_path> <model_tag> [task_id ...]

Examples:
    python s1_extract_spectra.py D:/models/Qwen3-8B qwen3_8b
    python s1_extract_spectra.py D:/models/Qwen3-8B qwen3_8b TR01
    python s1_extract_spectra.py D:/models/gemma-3-4b-it gemma3_4b
    python s1_extract_spectra.py D:/models/Ministral-8B-Instruct-2410 ministral8b

For each prompt × each layer × each head, computes:
    - erank       : entropy-based effective rank (non-integer)
    - stable_rank : sum(sigma_i^2) / sigma_max^2
    - energy_top1 : fraction of total singular-value mass in top-1 singular value
    - energy_top5 : fraction in top-5
    - diag_conc   : diagonal-band concentration (width=5), measures locality
    - mean_dist   : mean token distance weighted by attention weights

All raw attention matrices are discarded after metric computation (no N×N storage).
Output: results/spectral/spectral_{model_tag}.csv
"""
import sys
import csv
import gc
import math
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if len(sys.argv) < 3:
    print("Usage: python s1_extract_spectra.py <model_path> <model_tag> [task_id ...]")
    sys.exit(1)

MODEL_PATH = sys.argv[1]
MODEL_TAG  = sys.argv[2]
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = PROJECT_ROOT / "prompts"
OUT_CSV    = PROJECT_ROOT / "results" / "spectral" / f"spectral_{MODEL_TAG}.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_TASK_IDS = ["TR01","TR02","TR05","TR08","BZ01","BZ02","BZ05","BZ09",
                    "TC01","TC02","TC08","TC10"]
TASK_IDS = sys.argv[3:] if len(sys.argv) > 3 else DEFAULT_TASK_IDS
DOMAIN_MAP = {
    "TR01":"travel","TR02":"travel","TR05":"travel","TR08":"travel",
    "BZ01":"business","BZ02":"business","BZ05":"business","BZ09":"business",
    "TC01":"technical","TC02":"technical","TC08":"technical","TC10":"technical",
}

DIAG_WIDTH = 5   # half-width for diagonal concentration


def compute_spectral_metrics(A: torch.Tensor) -> dict:
    """
    A: (seq_len, seq_len) float32 attention matrix for one head.
    Returns a dict of scalar metrics.
    """
    N = A.shape[0]
    A_f = A.float()

    # SVD — attention matrices are not square-symmetric, use full SVD
    try:
        sigma = torch.linalg.svdvals(A_f)   # shape (min(N,N),) = (N,)
    except Exception:
        return {}

    sigma = sigma.clamp(min=0.0)
    total_sigma = sigma.sum().item()
    if total_sigma < 1e-12:
        return {}

    # --- erank (entropy-based effective rank) ---
    p = sigma / total_sigma
    p_nz = p[p > 1e-12]
    entropy = -(p_nz * torch.log(p_nz)).sum().item()
    erank = math.exp(entropy)

    # --- stable rank ---
    sigma2 = (sigma ** 2)
    stable_rank = (sigma2.sum() / sigma2[0].clamp(min=1e-12)).item()

    # --- top-k energy fraction ---
    cum = sigma2.cumsum(dim=0)
    total2 = sigma2.sum().item()
    energy_top1 = (sigma2[:1].sum() / total2).item() if total2 > 0 else 0.0
    energy_top5 = (sigma2[:5].sum() / total2).item() if N >= 5 and total2 > 0 else energy_top1

    # --- diagonal concentration (locality) ---
    # Build index grid
    rows = torch.arange(N, device=A.device).unsqueeze(1).expand(N, N)
    cols = torch.arange(N, device=A.device).unsqueeze(0).expand(N, N)
    dist = (rows - cols).abs()
    band_mask = (dist <= DIAG_WIDTH).float()
    diag_conc = (A_f * band_mask).sum().item() / A_f.sum().clamp(min=1e-12).item()

    # --- mean attention distance ---
    mean_dist = (A_f * dist.float()).sum().item() / A_f.sum().clamp(min=1e-12).item()

    return {
        "erank":       round(erank, 4),
        "stable_rank": round(stable_rank, 4),
        "energy_top1": round(energy_top1, 4),
        "energy_top5": round(energy_top5, 4),
        "diag_conc":   round(diag_conc, 4),
        "mean_dist":   round(mean_dist, 4),
        "seq_len":     N,
    }


def load_tokenizer(model_path: str):
    kwargs = {}
    if "mistral" in model_path.lower() or "ministral" in model_path.lower():
        kwargs["fix_mistral_regex"] = True
    return AutoTokenizer.from_pretrained(model_path, **kwargs)


def main():
    print(f"Model : {MODEL_PATH}")
    print(f"Tag   : {MODEL_TAG}")
    print(f"Output: {OUT_CSV}")

    tok = load_tokenizer(MODEL_PATH)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    if torch.cuda.is_available():
        print(f"VRAM  : {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    fields = [
        "task_id", "domain", "model_tag",
        "layer", "head", "seq_len",
        "erank", "stable_rank", "energy_top1", "energy_top5",
        "diag_conc", "mean_dist",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()

        for task_id in TASK_IDS:
            prompt_path = PROMPT_DIR / f"{task_id}__FULL.txt"
            if not prompt_path.exists():
                print(f"  [SKIP] {task_id} — prompt file not found")
                continue

            prompt_text = prompt_path.read_text(encoding="utf-8")
            inputs = tok(prompt_text, return_tensors="pt").to(DEVICE)
            seq_len = inputs["input_ids"].shape[1]
            domain  = DOMAIN_MAP.get(task_id, "unknown")

            print(f"  [{task_id}] seq_len={seq_len} ...", end=" ", flush=True)

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            # outputs.attentions: tuple of (1, n_heads, seq_len, seq_len) per layer
            n_layers = len(outputs.attentions)
            n_heads  = outputs.attentions[0].shape[1]
            rows_written = 0

            for layer_idx, layer_attn in enumerate(outputs.attentions):
                # layer_attn: (1, n_heads, seq_len, seq_len)
                attn_np = layer_attn[0].cpu()   # (n_heads, seq_len, seq_len)

                for head_idx in range(n_heads):
                    A = attn_np[head_idx]        # (seq_len, seq_len)
                    metrics = compute_spectral_metrics(A)
                    if not metrics:
                        continue

                    row = {
                        "task_id":   task_id,
                        "domain":    domain,
                        "model_tag": MODEL_TAG,
                        "layer":     layer_idx,
                        "head":      head_idx,
                    }
                    row.update(metrics)
                    writer.writerow(row)
                    rows_written += 1

                fout.flush()

            print(f"layers={n_layers} heads={n_heads} rows={rows_written}")

            del outputs, attn_np
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(f"\n✅ Done → {OUT_CSV}")


if __name__ == "__main__":
    main()
