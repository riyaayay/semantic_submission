"""
v10 set-level judge: a Jev-style "System One" decision model over each S1
entity's candidate set.

Design ideas follow the open-source reconstruction of TypeSafe AI's Jev,
kyegomez/open-jev (Apache-2.0, https://github.com/kyegomez/open-jev),
re-implemented here for entity resolution (no code copied):

- Bidirectional transformer over the whole "state" (no causal mask). The state
  is one S1 entity token plus its top-m candidate tokens; each candidate token
  embeds its v9 pair features and v9 score.
- Graph consistency: candidate-to-candidate similarity features (are two
  targets duplicates of each other, or a sister company / different address?)
  enter every attention layer as learned additive biases (Graphormer-style),
  so the decision for one target depends on the others.
- Typed query slots read decisions out in one parallel pass: a per-candidate
  "is this a match?" slot and an entity-level "does this entity have any
  match?" slot (targets singleton false merges).
- Evidential head: each answer is a Beta(alpha, beta) whose mean is the
  probability and whose total evidence is the confidence.
- RLCD-style calibrated loss: expected NLL under the Beta, Brier score,
  evidential regulariser, and a consistency term between two augmented passes.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpu import require_cuda

M_CANDIDATES = 32   # top-m candidates per S1 read by the judge (99.57% of true matches)

# candidate-to-candidate (target-target) edge features, computed by tt_edge_features
EDGE_FEATURE_NAMES = [
    "name_ratio", "name_token_sort_ratio", "name_tfidf_cosine", "name_char3_cosine", "name_soft_idf_min",
    "name_unmatched_count", "name_skel_ratio", "addr_tfidf_cosine", "addr_soft_idf_min", "addr_nums_jaccard",
    "addr_nums_conflict", "name_sem_cos", "same_source",
]


def tt_edge_features(prep_targets, voc, groups: np.ndarray, t_ids, t_name_emb: Optional[np.ndarray] = None,
                     rows_per_chunk: int = 10000, pair_batch: int = 200_000) -> np.ndarray:
    """Graph-consistency edges: features between every two candidates of the same S1.

    groups: (N, m) indices into t_ids (-1 = padding). t_name_emb: unit name embeddings
    aligned with t_ids (for semantic target-target cosine). Returns (N, m, m, Fe) float16,
    symmetric, zero on the diagonal and for padding.
    """
    import polars as pl
    from gpu_features import FEATURE_NAMES_V8, encode_bank, pair_features_v8

    dev = require_cuda()
    N, m = groups.shape
    cols = [FEATURE_NAMES_V8.index(n) for n in EDGE_FEATURE_NAMES[:11]]
    out = np.zeros((N, m, m, len(EDGE_FEATURE_NAMES)), dtype=np.float16)
    iu, ju = np.triu_indices(m, k=1)
    is_s3 = np.array([t.startswith("S3") for t in t_ids])
    # Restrict the target table ONCE to the records these groups reference: joining every chunk
    # against the full country table (millions of rows) dominated the run time.
    used = np.unique(groups[groups >= 0])
    prep_targets = prep_targets.join(pl.DataFrame({"target_id": [t_ids[i] for i in used]}), on="target_id", how="semi")
    for lo in range(0, N, rows_per_chunk):
        g = groups[lo:lo + rows_per_chunk]
        uniq = np.unique(g[g >= 0])
        if len(uniq) == 0:
            continue
        local = np.full(len(t_ids), -1, dtype=np.int64)
        local[uniq] = np.arange(len(uniq))
        order = pl.DataFrame({"target_id": [t_ids[i] for i in uniq], "_o": np.arange(len(uniq))})
        frame = order.join(prep_targets, on="target_id", how="left").sort("_o").drop("_o")
        bank = encode_bank(frame, voc, "target_id", pattern_side=True)
        a, b = g[:, iu], g[:, ju]                                    # (rows, P) global target index
        ok = (a >= 0) & (b >= 0)
        r_idx, p_idx = np.nonzero(ok)
        ga, gb = a[ok], b[ok]
        feats = np.empty((len(ga), len(EDGE_FEATURE_NAMES)), dtype=np.float32)
        for s in range(0, len(ga), pair_batch):
            ia = torch.as_tensor(local[ga[s:s + pair_batch]], device=dev)
            ib = torch.as_tensor(local[gb[s:s + pair_batch]], device=dev)
            feats[s:s + pair_batch, :11] = pair_features_v8(bank, bank, ia, ib, edge_only=True)[:, cols].cpu().numpy()
        if t_name_emb is not None:
            feats[:, 11] = (t_name_emb[ga].astype(np.float32) * t_name_emb[gb].astype(np.float32)).sum(1)
        feats[:, 12] = (is_s3[ga] == is_s3[gb]).astype(np.float32)
        f16 = feats.astype(np.float16)
        rows = r_idx + lo
        out[rows, iu[p_idx], ju[p_idx]] = f16
        out[rows, ju[p_idx], iu[p_idx]] = f16
        del bank
        torch.cuda.empty_cache()
    return out


RANK_FEATURES = ["name_ratio", "name_soft_idf_mean", "addr_soft_idf_mean", "addr_tfidf_cosine", "name_sem_cos"]


def topm_groups(ia: np.ndarray, X, feature_names, m: int = M_CANDIDATES) -> np.ndarray:
    """Label-free top-m candidates per S1. ia: S1 index per pair (non-decreasing,
    0..n_s1-1); X: (n_pairs, n_feat) array/memmap. Returns (n_s1, m) pair indices, -1 padded."""
    h = np.zeros(len(ia), dtype=np.float32)
    for name in RANK_FEATURES:
        h += np.asarray(X[:, feature_names.index(name)], dtype=np.float32)
    n_s1 = int(ia.max()) + 1 if len(ia) else 0
    order = np.lexsort((-h, ia))
    sizes = np.bincount(ia, minlength=n_s1)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    rank = np.empty(len(ia), np.int64)
    rank[order] = np.arange(len(ia)) - np.repeat(starts, sizes)
    groups = np.full((n_s1, m), -1, dtype=np.int64)
    keep = rank < m
    groups[ia[keep], rank[keep]] = np.nonzero(keep)[0]
    return groups


def make_judge_data(groups: np.ndarray, X, E: np.ndarray, p: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                    y: Optional[np.ndarray] = None) -> "JudgeData":
    """Node tokens = standardized v9 features + v9 score (logit and prob) + slot rank."""
    mask = groups >= 0
    gi = np.where(mask, groups, 0)
    Xg = np.asarray(X[gi.ravel()], dtype=np.float32).reshape(*gi.shape, -1)
    pg = p[gi]
    logit = np.log(np.clip(pg, 1e-6, 1 - 1e-6) / np.clip(1 - pg, 1e-6, 1))
    m = groups.shape[1]
    rank = np.broadcast_to(np.arange(m, dtype=np.float32) / m, mask.shape)
    nodes = np.concatenate([(Xg - mu) / sd, logit[..., None], pg[..., None], rank[..., None]], -1) * mask[..., None]
    yy = ya = None
    if y is not None:
        yy = torch.as_tensor((y[gi] * mask).astype(np.float32))
        ya = (yy.sum(1) > 0).float()
    return JudgeData(torch.as_tensor(nodes, dtype=torch.float16), torch.as_tensor(np.asarray(E)),
                     torch.as_tensor(mask), yy, ya)


def save_judge(path: str, model: "SetJudge", mu, sd, tau: float, extra: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "cfg": model.cfg.__dict__, "mu": mu, "sd": sd,
                "tau": tau, "m": M_CANDIDATES, **extra}, path)


def load_judge(path: str):
    dev = require_cuda()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = SetJudge(JudgeConfig(**ck["cfg"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(dev).eval(), ck


@dataclass
class JudgeConfig:
    n_node_features: int
    n_edge_features: int = len(EDGE_FEATURE_NAMES)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1


class BiasedEncoderLayer(nn.Module):
    """Pre-norm transformer layer whose attention takes a per-head additive bias."""

    def __init__(self, cfg: JudgeConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dk = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.n1 = nn.LayerNorm(cfg.d_model)
        self.n2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout),
                                nn.Linear(cfg.d_ff, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, bias):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), dropout_p=0.0)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(self.n2(x)))


class SetJudge(nn.Module):
    def __init__(self, cfg: JudgeConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.node_in = nn.Sequential(nn.Linear(cfg.n_node_features, d), nn.GELU(), nn.Linear(d, d))
        self.entity_tok = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.edge_in = nn.Sequential(nn.Linear(cfg.n_edge_features, 32), nn.GELU(),
                                     nn.Linear(32, cfg.n_heads * cfg.n_layers))
        self.layers = nn.ModuleList([BiasedEncoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(d)
        # typed query slots
        self.match_slot = nn.Linear(d, 2)                 # evidence (pos, neg) per candidate
        self.any_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.any_attn = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        self.any_slot = nn.Linear(d, 2)                   # evidence that the entity has any match

    def forward(self, nodes, edges, mask):
        """nodes (B,m,Fn), edges (B,m,m,Fe), mask (B,m) bool valid -> dict of Beta parameters."""
        B, m, _ = nodes.shape
        L, H = self.cfg.n_layers, self.cfg.n_heads
        x = torch.cat([self.entity_tok.expand(B, 1, -1), self.node_in(nodes)], 1)      # (B, m+1, d)
        eb = self.edge_in(edges).view(B, m, m, L, H).permute(3, 0, 4, 1, 2)            # (L, B, H, m, m)
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], 1)
        pad = torch.zeros(B, 1, m + 1, m + 1, device=nodes.device, dtype=nodes.dtype)
        pad = pad.masked_fill(~valid[:, None, None, :], float("-inf"))
        for li, layer in enumerate(self.layers):
            bias = pad.expand(B, H, m + 1, m + 1).clone()
            bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + eb[li]
            x = layer(x, bias)
        x = self.norm(x)
        ev = F.softplus(self.match_slot(x[:, 1:]))                                     # (B, m, 2)
        q = self.any_query.expand(B, 1, -1)
        a, _ = self.any_attn(q, x, x, key_padding_mask=~valid)
        ev_any = F.softplus(self.any_slot(a[:, 0]))                                     # (B, 2)
        return {"alpha": ev[..., 0] + 1, "beta": ev[..., 1] + 1,
                "alpha_any": ev_any[:, 0] + 1, "beta_any": ev_any[:, 1] + 1}


def beta_mean(alpha, beta):
    return alpha / (alpha + beta)


def rlcd_loss(out, out_aug, y, y_any, mask, w_brier=1.0, w_reg=0.05, w_cons=0.5, w_any=0.5):
    """Calibrated-decision loss (RLCD-style): expected NLL under Beta + Brier +
    evidential regulariser on wrong-direction evidence + consistency between two
    augmented passes, for candidate slots (masked) and the entity slot."""
    def terms(a, b, t):
        s = a + b
        nll = t * (torch.digamma(s) - torch.digamma(a)) + (1 - t) * (torch.digamma(s) - torch.digamma(b))
        p = a / s
        brier = (p - t) ** 2
        # evidence pointing the wrong way should vanish (Sensoy et al. evidential regulariser, binary form)
        wrong = t * (b - 1) + (1 - t) * (a - 1)
        return nll + w_brier * brier + w_reg * wrong / s, p

    tm, pm = terms(out["alpha"], out["beta"], y)
    ta, pa = terms(out["alpha_any"], out["beta_any"], y_any)
    _, pm2 = terms(out_aug["alpha"], out_aug["beta"], y)
    cons = ((pm - pm2) ** 2 * mask).sum() / mask.sum()
    loss_m = (tm * mask).sum() / mask.sum()
    return loss_m + w_any * ta.mean() + w_cons * cons


@dataclass
class JudgeData:
    nodes: torch.Tensor     # (N, m, Fn) float16, CPU
    edges: torch.Tensor     # (N, m, m, Fe) float16, CPU
    mask: torch.Tensor      # (N, m) bool
    y: Optional[torch.Tensor] = None       # (N, m) float
    y_any: Optional[torch.Tensor] = None   # (N,) float

    def subset(self, idx):
        return JudgeData(self.nodes[idx], self.edges[idx], self.mask[idx],
                         None if self.y is None else self.y[idx], None if self.y_any is None else self.y_any[idx])


def _augment(nodes, edges, noise=0.02):
    return nodes + noise * torch.randn_like(nodes), edges + noise * torch.randn_like(edges)


def train_judge(train: JudgeData, val: JudgeData, cfg: JudgeConfig, epochs: int = 30, batch: int = 512,
                lr: float = 2e-3, patience: int = 4, seed: int = 42, log=print) -> SetJudge:
    """Train on the GPU with early stopping on validation loss; returns the best model."""
    dev = require_cuda()
    torch.manual_seed(seed)
    model = SetJudge(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(train.mask) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(train.mask))
        tot = 0.0
        for lo in range(0, len(perm), batch):
            b = perm[lo:lo + batch]
            n, e = train.nodes[b].to(dev).float(), train.edges[b].to(dev).float()
            mk, y, ya = train.mask[b].to(dev), train.y[b].to(dev), train.y_any[b].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(n, e, mk)
                out2 = model(*_augment(n, e), mk)
            loss = rlcd_loss({k: v.float() for k, v in out.items()}, {k: v.float() for k, v in out2.items()},
                             y, ya, mk.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        vl = evaluate_loss(model, val, batch)
        log(f"    judge epoch {ep + 1:2d}: train loss {tot / len(perm):.4f}  val loss {vl:.4f}  ({time.time() - t0:.0f}s)")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def evaluate_loss(model, data: JudgeData, batch: int = 1024) -> float:
    dev = require_cuda()
    model.eval()
    tot = 0.0
    for lo in range(0, len(data.mask), batch):
        n, e = data.nodes[lo:lo + batch].to(dev).float(), data.edges[lo:lo + batch].to(dev).float()
        mk = data.mask[lo:lo + batch].to(dev)
        out = model(n, e, mk)
        tot += rlcd_loss(out, out, data.y[lo:lo + batch].to(dev), data.y_any[lo:lo + batch].to(dev),
                         mk.float(), w_cons=0.0).item() * len(n)
    return tot / len(data.mask)


@torch.no_grad()
def predict_judge(model, data: JudgeData, batch: int = 1024) -> Dict[str, np.ndarray]:
    """Per-candidate match probability and confidence, plus entity-level any-match probability."""
    dev = require_cuda()
    model.eval()
    p, conf, p_any = [], [], []
    for lo in range(0, len(data.mask), batch):
        n, e = data.nodes[lo:lo + batch].to(dev).float(), data.edges[lo:lo + batch].to(dev).float()
        out = model(n, e, data.mask[lo:lo + batch].to(dev))
        s = out["alpha"] + out["beta"]
        p.append((out["alpha"] / s).cpu())
        conf.append((1 - 2 / s).cpu())
        p_any.append(beta_mean(out["alpha_any"], out["beta_any"]).cpu())
    return {"p": torch.cat(p).numpy(), "conf": torch.cat(conf).numpy(), "p_any": torch.cat(p_any).numpy()}
