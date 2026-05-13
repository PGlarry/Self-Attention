"""
s2_approximate.py — Module 2: Structured Matrix Approximation Benchmark
Usage:
    python s2_approximate.py <model_path> <model_tag> [task_id ...]

Re-runs forward passes on the same 12 FULL prompts, but this time computes
approximation errors for four structural families at multiple parameter levels.
Metrics are computed on-the-fly; raw matrices are not saved.

Approximation families:
    lowrank  : truncated SVD at rank r = 2, 4, 8, 16
    banded   : retain diagonal band of half-width w = 2, 5, 10, 20, then renormalize rows
    sparse   : keep top-k% of entries per row, renormalize (k = 5, 10, 20, 40)
    monarch  : block-diagonal approximation with block size b = 2, 4, 8 (simplified Monarch)

Error metrics per (task, layer, head, method, param):
    frob_rel : ||A - A_approx||_F / ||A||_F
    kl_mean  : mean row-wise KL(A_row || A_approx_row)  [after clamping negatives]

Output: results/approx/approx_{model_tag}.csv
"""
import sys
import csv
import gc
import math
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if len(sys.argv) < 3:
    print("Usage: python s2_approximate.py <model_path> <model_tag> [task_id ...]")
    sys.exit(1)

MODEL_PATH = sys.argv[1]
MODEL_TAG  = sys.argv[2]
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = PROJECT_ROOT / "prompts"
OUT_CSV    = PROJECT_ROOT / "results" / "approx" / f"approx_{MODEL_TAG}.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_TASK_IDS = ["TR01","TR02","TR05","TR08","BZ01","BZ02","BZ05","BZ09",
                    "TC01","TC02","TC08","TC10"]
TASK_IDS = sys.argv[3:] if len(sys.argv) > 3 else DEFAULT_TASK_IDS
DOMAIN_MAP = {
    "TR01":"travel","TR02":"travel","TR05":"travel","TR08":"travel",
    "BZ01":"business","BZ02":"business","BZ05":"business","BZ09":"business",
    "TC01":"technical","TC02":"technical","TC08":"technical","TC10":"technical",
}

# Approximation parameters
LOWRANK_RANKS  = [2, 4, 8, 16]
BANDED_WIDTHS  = [2, 5, 10, 20]     # half-width (total band = 2w+1)
SPARSE_TOPK    = [5, 10, 20, 40]    # percentage of entries per row to keep
MONARCH_BLOCKS = [2, 4, 8]          # number of blocks (block size = N // n_blocks)


# ── Approximation functions ───────────────────────────────────────────────────

def approx_lowrank(A: torch.Tensor, rank: int) -> torch.Tensor:
    """Truncated SVD approximation. Result may have negative values; clamp & renorm."""
    U, S, Vh = torch.linalg.svd(A.float(), full_matrices=False)
    r = min(rank, S.shape[0])
    return lowrank_from_svd(U, S, Vh, r)


def lowrank_from_svd(U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, rank: int) -> torch.Tensor:
    """Build a row-normalized nonnegative low-rank approximation from cached SVD factors."""
    r = min(rank, S.shape[0])
    A_approx = (U[:, :r] * S[:r]) @ Vh[:r, :]
    A_approx = A_approx.clamp(min=0.0)
    row_sums = A_approx.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return A_approx / row_sums


def approx_banded(A: torch.Tensor, half_width: int) -> torch.Tensor:
    """Keep entries within half_width of diagonal, zero out rest, renormalize rows."""
    N = A.shape[0]
    rows = torch.arange(N, device=A.device).unsqueeze(1)
    cols = torch.arange(N, device=A.device).unsqueeze(0)
    mask = ((rows - cols).abs() <= half_width).float()
    A_approx = A.float() * mask
    row_sums = A_approx.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return A_approx / row_sums


def approx_sparse(A: torch.Tensor, topk_pct: int) -> torch.Tensor:
    """Keep top-k% entries per row (by value), zero rest, renormalize."""
    N = A.shape[0]
    k = max(1, int(math.ceil(N * topk_pct / 100.0)))
    A_f = A.float()
    topk_vals, topk_idx = torch.topk(A_f, k=min(k, N), dim=1)
    A_approx = torch.zeros_like(A_f)
    A_approx.scatter_(1, topk_idx, topk_vals)
    row_sums = A_approx.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return A_approx / row_sums


def approx_monarch(A: torch.Tensor, n_blocks: int) -> torch.Tensor:
    """
    Simplified Monarch: block-diagonal approximation.
    Partition the N×N matrix into n_blocks×n_blocks submatrices;
    keep only the block-diagonal blocks, zero out off-diagonal blocks, renormalize.
    When N is not divisible by n_blocks, the last block absorbs the remainder.
    """
    N = A.shape[0]
    block_size = N // n_blocks
    if block_size < 1:
        return A.clone()
    A_f = A.float()
    mask = torch.zeros(N, N, device=A.device)
    for b in range(n_blocks):
        r_start = b * block_size
        r_end   = r_start + block_size if b < n_blocks - 1 else N
        mask[r_start:r_end, r_start:r_end] = 1.0
    A_approx = A_f * mask
    row_sums = A_approx.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return A_approx / row_sums


# ── Error metrics ────────────────────────────────────────────────────────────

def frob_rel(A: torch.Tensor, A_approx: torch.Tensor) -> float:
    denom = A.float().norm(p="fro").item()
    if denom < 1e-12:
        return 0.0
    return ((A.float() - A_approx).norm(p="fro") / denom).item()


def kl_mean(A: torch.Tensor, A_approx: torch.Tensor, eps: float = 1e-12) -> float:
    """Row-wise KL(A || A_approx), averaged over rows."""
    P = A.float().clamp(min=eps)
    Q = A_approx.float().clamp(min=eps)
    # normalize to ensure valid distributions
    P = P / P.sum(dim=1, keepdim=True)
    Q = Q / Q.sum(dim=1, keepdim=True)
    kl_rows = (P * (P / Q).log()).sum(dim=1)   # (N,)
    return kl_rows.mean().item()


def load_tokenizer(model_path: str):
    kwargs = {}
    if "mistral" in model_path.lower() or "ministral" in model_path.lower():
        kwargs["fix_mistral_regex"] = True
    return AutoTokenizer.from_pretrained(model_path, **kwargs)


# ── Main ─────────────────────────────────────────────────────────────────────

def compute_all_errors(A: torch.Tensor) -> list[dict]:
    """Return list of error dicts for all (method, param) combinations."""
    records = []
    N = A.shape[0]

    U, S, Vh = torch.linalg.svd(A.float(), full_matrices=False)
    for r in LOWRANK_RANKS:
        if r >= N:
            continue
        Aapp = lowrank_from_svd(U, S, Vh, r)
        records.append({"method": "lowrank", "param": r,
                         "frob_rel": frob_rel(A, Aapp),
                         "kl_mean":  kl_mean(A, Aapp)})

    for w in BANDED_WIDTHS:
        Aapp = approx_banded(A, w)
        records.append({"method": "banded", "param": w,
                         "frob_rel": frob_rel(A, Aapp),
                         "kl_mean":  kl_mean(A, Aapp)})

    for pct in SPARSE_TOPK:
        Aapp = approx_sparse(A, pct)
        records.append({"method": "sparse_topk", "param": pct,
                         "frob_rel": frob_rel(A, Aapp),
                         "kl_mean":  kl_mean(A, Aapp)})

    for nb in MONARCH_BLOCKS:
        if nb > N:
            continue
        Aapp = approx_monarch(A, nb)
        records.append({"method": "monarch", "param": nb,
                         "frob_rel": frob_rel(A, Aapp),
                         "kl_mean":  kl_mean(A, Aapp)})

    return records


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
        "task_id", "domain", "model_tag", "layer", "head", "seq_len",
        "method", "param", "frob_rel", "kl_mean",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()

        for task_id in TASK_IDS:
            prompt_path = PROMPT_DIR / f"{task_id}__FULL.txt"
            if not prompt_path.exists():
                print(f"  [SKIP] {task_id}")
                continue

            prompt_text = prompt_path.read_text(encoding="utf-8")
            inputs  = tok(prompt_text, return_tensors="pt").to(DEVICE)
            seq_len = inputs["input_ids"].shape[1]
            domain  = DOMAIN_MAP.get(task_id, "unknown")

            print(f"  [{task_id}] seq_len={seq_len} ...", end=" ", flush=True)

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            n_layers = len(outputs.attentions)
            n_heads  = outputs.attentions[0].shape[1]
            rows_written = 0

            for layer_idx, layer_attn in enumerate(outputs.attentions):
                attn_cpu = layer_attn[0].cpu()  # (n_heads, seq_len, seq_len)

                for head_idx in range(n_heads):
                    A = attn_cpu[head_idx].float()
                    error_records = compute_all_errors(A)

                    for rec in error_records:
                        row = {
                            "task_id":   task_id,
                            "domain":    domain,
                            "model_tag": MODEL_TAG,
                            "layer":     layer_idx,
                            "head":      head_idx,
                            "seq_len":   seq_len,
                            "method":    rec["method"],
                            "param":     rec["param"],
                            "frob_rel":  round(rec["frob_rel"], 6),
                            "kl_mean":   round(rec["kl_mean"], 6),
                        }
                        writer.writerow(row)
                        rows_written += 1

                fout.flush()

            print(f"layers={n_layers} heads={n_heads} rows={rows_written}")

            del outputs, attn_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(f"\n✅ Done → {OUT_CSV}")


if __name__ == "__main__":
    main()
