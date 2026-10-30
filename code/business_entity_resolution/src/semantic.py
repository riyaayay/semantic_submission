"""
Multilingual semantic embeddings (v9 layer).

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(Apache-2.0, 118M parameters, 50+ languages, 384-dim, audited in
runs/neural_retrieval_pilot). It maps text with similar meaning to nearby
vectors across languages and scripts ("Boulangerie" ~ "Bakery",
"मॉडर्न फाइनेंस" ~ "Modern Finance"), so it is fed the ORIGINAL text, not the
Latin-folded cleaned text.

Embeddings are L2-normalized float16, computed on the GPU (no CPU fallback).
"""

import os
from typing import List, Optional

import numpy as np
import torch

from gpu import require_cuda

_HERE = os.path.dirname(os.path.abspath(__file__))
# Shipped inside the code folder (models/semantic_minilm) so the package reproduces offline.
DEFAULT_MODEL_DIR = os.environ.get("SEMANTIC_MODEL_DIR", os.path.join(_HERE, "..", "models", "semantic_minilm"))
DIM = 384
_model = None


def load_model(path: Optional[str] = None):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        dev = require_cuda()
        _model = SentenceTransformer(os.path.abspath(path or DEFAULT_MODEL_DIR), device=str(dev))
        _model.half()
        _model.max_seq_length = 64
    return _model


def embed(texts: List[str], batch_size: int = 1024) -> np.ndarray:
    """(N, 384) float16 unit vectors. Empty strings map to the zero vector."""
    model = load_model()
    out = np.zeros((len(texts), DIM), dtype=np.float16)
    idx = [i for i, t in enumerate(texts) if t and t.strip()]
    if idx:
        # sort by length so batches are padded tightly (big GPU speedup)
        idx.sort(key=lambda i: len(texts[i]))
        with torch.inference_mode():
            for lo in range(0, len(idx), batch_size * 16):
                part = idx[lo:lo + batch_size * 16]
                v = model.encode([texts[i] for i in part], batch_size=batch_size, convert_to_tensor=True,
                                 normalize_embeddings=True, show_progress_bar=False)
                out[part] = v.to(torch.float16).cpu().numpy()
    return out


def topk_search(q: np.ndarray, base: np.ndarray, k: int, base_chunk: int = 500_000, q_chunk: int = 2048) -> np.ndarray:
    """Exact cosine top-k of each query row against base rows (unit fp16 vectors), on the GPU.

    Score tiles are at most (q_chunk x base_chunk) fp16 = 2 GB.
    """
    dev = require_cuda()
    k = min(k, len(base))
    best_s = torch.full((len(q), k), -2.0, device=dev, dtype=torch.float16)
    best_i = torch.zeros((len(q), k), device=dev, dtype=torch.long)
    for lo in range(0, len(base), base_chunk):
        B = torch.as_tensor(base[lo:lo + base_chunk], device=dev)
        kk = min(k, len(B))
        for qlo in range(0, len(q), q_chunk):
            Q = torch.as_tensor(q[qlo:qlo + q_chunk], device=dev)
            s, i = (Q @ B.T).topk(kk, dim=1)
            cs = torch.cat([best_s[qlo:qlo + q_chunk], s], 1)
            ci = torch.cat([best_i[qlo:qlo + q_chunk], i + lo], 1)
            top = cs.topk(k, dim=1)
            best_s[qlo:qlo + q_chunk] = top.values
            best_i[qlo:qlo + q_chunk] = torch.gather(ci, 1, top.indices)
        del B
    out = best_i.cpu().numpy()
    del best_s, best_i
    torch.cuda.empty_cache()
    return out


def pair_cosine(a: np.ndarray, b: np.ndarray, ia: np.ndarray, ib: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """cos(a[ia[k]], b[ib[k]]) for every pair, on the GPU."""
    dev = require_cuda()
    A = torch.as_tensor(a, device=dev)
    B = torch.as_tensor(b, device=dev)
    out = np.empty(len(ia), dtype=np.float32)
    for lo in range(0, len(ia), chunk):
        x = A[torch.as_tensor(ia[lo:lo + chunk], device=dev, dtype=torch.long)]
        y = B[torch.as_tensor(ib[lo:lo + chunk], device=dev, dtype=torch.long)]
        out[lo:lo + chunk] = (x.float() * y.float()).sum(1).cpu().numpy()
    del A, B
    torch.cuda.empty_cache()
    return out
