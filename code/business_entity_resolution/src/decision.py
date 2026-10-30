"""
Turning pair probabilities into per-entity match sets.

Two policies, both enforcing that a target record joins at most one S1 entity
(the one that scores it highest):

- threshold: keep every pair with p >= tau.
- expected F-beta: per S1 entity, pick the prediction set that maximizes the
  expected F-beta score under the model's probabilities, including the empty
  set (which scores 1.0 exactly when the entity is a singleton).
"""

from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


def _exclusive(pairs: Sequence[Tuple[str, str]], scores: np.ndarray) -> Dict[str, List[Tuple[str, float]]]:
    """Keeps each target only under its highest-scoring S1; groups by S1."""
    best: Dict[str, Tuple[str, float]] = {}
    for (sid, tid), p in zip(pairs, scores):
        p = float(p)
        cur = best.get(tid)
        if cur is None or p > cur[1] or (p == cur[1] and sid < cur[0]):
            best[tid] = (sid, p)
    by_s1: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for tid, (sid, p) in best.items():
        by_s1[sid].append((tid, p))
    return by_s1


def assign_threshold(pairs: Sequence[Tuple[str, str]], scores: np.ndarray, tau: float) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    for sid, cands in _exclusive(pairs, scores).items():
        out[sid] = {tid for tid, p in cands if p >= tau}
    return out


def best_expected_f_set(probs: np.ndarray, beta2: float = 0.25, floor: float = 0.0) -> int:
    """How many of the (descending) probabilities to predict to maximize E[F_beta].

    Treats candidates as independent Bernoullis. With P0 = prod(1 - p) the
    chance the entity is a singleton, predicting nothing earns P0. Predicting
    the top k earns, conditioned on at least one true match, roughly
        (1 + beta2) * E[TP_k] / (beta2 * E[|T|] + k)
    with E[TP_k] = sum(p[:k]) and E[|T|] = sum(p) (ratio of expectations),
    both divided by (1 - P0) for the conditioning; the (1 - P0) factors cancel
    in the numerator, leaving the formula below. `floor` drops candidates with
    p below it before optimizing.
    """
    p = probs[probs >= floor] if floor > 0 else probs
    if p.size == 0:
        return 0
    p0 = float(np.prod(1.0 - p))
    cond = max(1.0 - p0, 1e-12)
    e_true = p.sum() / cond
    tp = np.cumsum(p) / cond
    k = np.arange(1, p.size + 1)
    ef = (1.0 - p0) * (1.0 + beta2) * tp / (beta2 * e_true + k)
    best_k = int(np.argmax(ef)) + 1
    return best_k if ef[best_k - 1] > p0 else 0


def em_prior_shift(probs: np.ndarray, train_prior: float, iters: int = 100, tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    """Label-free recalibration for a country with a different match rate.

    Saerens, Latinne & Decaestecker (2002): under label shift p(x|y) is fixed
    and only the prior pi = P(y=1) changes. With r = pi_new / pi_train and
    s = (1 - pi_new) / (1 - pi_train), the adjusted posterior is
        p' = r p / (r p + s (1 - p)),
    and EM alternates that with pi_new = mean(p') until it converges.
    Returns (adjusted probabilities, estimated prior).
    """
    p = np.clip(probs.astype(np.float64), 1e-7, 1 - 1e-7)
    pi = float(p.mean())
    adj = p
    for _ in range(iters):
        r, s = pi / train_prior, (1 - pi) / (1 - train_prior)
        adj = r * p / (r * p + s * (1 - p))
        new = float(adj.mean())
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return adj, pi


def assign_expected_f(
    pairs: Sequence[Tuple[str, str]], scores: np.ndarray, beta2: float = 0.25, floor: float = 0.0
) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    for sid, cands in _exclusive(pairs, scores).items():
        cands.sort(key=lambda x: (-x[1], x[0]))
        k = best_expected_f_set(np.array([p for _, p in cands]), beta2, floor)
        out[sid] = {tid for tid, _ in cands[:k]}
    return out
