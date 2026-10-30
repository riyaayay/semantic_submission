"""
Bit-parallel string metrics on the GPU, batched over string pairs.

Strings are encoded as small-alphabet symbol matrices (see encode_strings).
For each "pattern" string A we precompute Peq[A, symbol] = bitmask of the
positions of that symbol in A, packed into W words of 63 bits. The "text"
string B is then consumed one character per step for the whole batch at once:

- LCS length (Hyyro 2004 / Allison-Dix):  U = S & Eq;  S = (S + U) | (S - U)
  LCS = number of zero bits of S within A's length. Multi-word addition carries
  are propagated explicitly, so long strings (addresses) work.
- Levenshtein distance (Myers 1999, Hyyro's formulation), single word
  (strings up to 63 symbols; longer strings are truncated).

These give rapidfuzz's fuzz.ratio (= 2*LCS / (len_a + len_b)) and
LCSseq.normalized_similarity (= LCS / max(len)) exactly on the encoded strings.
"""

import numpy as np
import torch

WORD_BITS = 63
MASK63 = (1 << 63) - 1

# Symbol alphabet after cleaning: a-z, 0-9, space; everything else -> OTHER.
_ALPHA = "abcdefghijklmnopqrstuvwxyz0123456789 "
N_SYMBOLS = len(_ALPHA) + 2          # + OTHER + PAD
SYM_OTHER = len(_ALPHA)
SYM_PAD = len(_ALPHA) + 1
_LUT = np.full(65536, SYM_OTHER, dtype=np.int8)   # code points above the BMP are capped to OTHER
for _i, _c in enumerate(_ALPHA):
    _LUT[ord(_c)] = _i


def encode_strings(strings, max_len: int):
    """CPU: list[str] -> (int8 symbols (N, max_len) padded with SYM_PAD, int16 lengths).

    One vectorized pass: pad/truncate, UTF-32 encode the concatenation, map
    code points through a lookup table.
    """
    fixed = [(s or "")[:max_len].ljust(max_len, "\0") for s in strings]
    lens = np.fromiter((min(len(s or ""), max_len) for s in strings), dtype=np.int16, count=len(strings))
    cps = np.frombuffer("".join(fixed).encode("utf-32-le"), dtype=np.uint32).reshape(len(strings), max_len)
    cps = np.minimum(cps, 65535)
    sym = _LUT[cps]
    sym[np.arange(max_len)[None, :] >= lens[:, None]] = SYM_PAD
    return sym, lens


def build_peq(sym: torch.Tensor, n_words: int) -> torch.Tensor:
    """GPU: (N, L) symbols -> (N, N_SYMBOLS, n_words) int64 position bitmasks. PAD gets no bits."""
    n, L = sym.shape
    assert L <= n_words * WORD_BITS
    dev = sym.device
    pos = torch.arange(L, device=dev)
    word = pos // WORD_BITS
    bit = torch.ones(L, dtype=torch.int64, device=dev) << (pos % WORD_BITS)
    # Each position sets one distinct bit, so summing bits per (symbol, word) slot
    # equals OR-ing them; scatter_add keeps memory O(n * L) (no one-hot tensor).
    peq = torch.zeros(n, N_SYMBOLS * n_words, dtype=torch.int64, device=dev)
    slot = sym.long() * n_words + word[None, :]
    peq.scatter_add_(1, slot, bit[None, :].expand(n, L))
    peq = peq.view(n, N_SYMBOLS, n_words)
    peq[:, SYM_PAD, :] = 0
    return peq


def popcount63(x: torch.Tensor) -> torch.Tensor:
    """Bit count of non-negative int64 values (bit 63 clear)."""
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    x = x + (x >> 32)
    return x & 0x7F


def _len_masks(len_a: torch.Tensor, n_words: int) -> torch.Tensor:
    """(P, W) masks with the first len_a bits set across words."""
    w = torch.arange(n_words, device=len_a.device)[None, :]
    bits = (len_a[:, None].long() - w * WORD_BITS).clamp(0, WORD_BITS)
    full = bits == WORD_BITS
    m = (torch.ones_like(bits) << bits) - 1
    return torch.where(full, torch.full_like(m, MASK63), m)


def lcs_length(peq_a: torch.Tensor, len_a: torch.Tensor, sym_b: torch.Tensor) -> torch.Tensor:
    """LCS length for each pair. peq_a: (P, S, W) gathered per pair; sym_b: (P, Lb)."""
    P, _, W = peq_a.shape
    S = torch.full((P, W), MASK63, dtype=torch.int64, device=peq_a.device)
    rows = torch.arange(P, device=peq_a.device)
    for j in range(sym_b.shape[1]):
        c = sym_b[:, j].long()
        if bool((c == SYM_PAD).all()):
            break
        eq = peq_a[rows, c]                     # (P, W); PAD rows are zero -> no-op step
        U = S & eq
        carry = torch.zeros(P, dtype=torch.int64, device=S.device)
        new = torch.empty_like(S)
        for w in range(W):
            t = S[:, w] + U[:, w] + carry       # < 2^64; bit 63 (sign) is the carry out
            carry = (t < 0).to(torch.int64)
            new[:, w] = (t & MASK63) | (S[:, w] & ~U[:, w])
        S = new
    zeros = popcount63(~S & _len_masks(len_a, W)).sum(1)
    return zeros


def levenshtein(peq_a: torch.Tensor, len_a: torch.Tensor, sym_b: torch.Tensor, len_b: torch.Tensor) -> torch.Tensor:
    """Levenshtein distance (single 63-bit word, Myers/Hyyro). peq_a: (P, S, 1)."""
    P = peq_a.shape[0]
    dev = peq_a.device
    rows = torch.arange(P, device=dev)
    m = len_a.long()
    mask = _len_masks(len_a, 1)[:, 0]
    top = (m - 1).clamp(min=0)
    Pv = mask.clone()
    Mv = torch.zeros(P, dtype=torch.int64, device=dev)
    score = m.clone()
    for j in range(sym_b.shape[1]):
        valid = j < len_b.long()
        if not bool(valid.any()):
            break
        eq = peq_a[rows, sym_b[:, j].long(), 0]
        Xv = eq | Mv
        Xh = ((((eq & Pv) + Pv) & MASK63) ^ Pv) | eq
        Ph = (Mv | ~(Xh | Pv)) & mask
        Mh = Pv & Xh
        d = ((Ph >> top) & 1) - ((Mh >> top) & 1)
        Ph = ((Ph << 1) | 1) & mask
        Mh = (Mh << 1) & mask
        nPv = (Mh | ~(Xv | Ph)) & mask
        nMv = Ph & Xv
        score = torch.where(valid & (m > 0), score + d, score)
        Pv = torch.where(valid, nPv, Pv)
        Mv = torch.where(valid, nMv, Mv)
    return torch.where(m == 0, len_b.long(), score)
