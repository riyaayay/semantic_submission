"""
GPU feature engine (feature set v8).

Records are encoded once into padded integer/float tensors (RecordBank); the
only CPU work is that per-record text encoding, done with vectorized Polars.
Every per-pair computation runs on the GPU in batches:

- string ratios: bit-parallel LCS / Levenshtein (gpu_strings.py)
- TF-IDF, char 3-gram and IDF-weighted Jaccard: batched sorted-set
  intersection via torch.searchsorted
- soft token alignment (IDF-weighted Monge-Elkan): token similarity from
  128-bit bigram bitmasks (Dice via popcount) plus an abbreviation rule
- group features: scatter reductions over each S1's candidates

IDF and name end-position statistics are fit per country on that country's
own target corpus (label-free), exactly as in v7.
"""

import math
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import torch

from features import compute_soundex
from gpu import require_cuda
from gpu_strings import MASK63, build_peq, encode_strings, lcs_length, levenshtein, popcount63

NT, AT, NC, NN, NK, NI = 8, 24, 64, 8, 8, 7       # max name tokens, addr tokens, char3 grams, numbers, skeleton tokens, initials prefixes
NAME_LEN, ADDR_LEN = 63, 189
SENT = torch.iinfo(torch.int64).max               # sorts after every real id
SOFT_MATCH = 0.6
ABBREV_SIM = 0.9

FEATURE_NAMES_V8 = [
    "name_ratio", "name_token_sort_ratio", "name_token_contain", "name_common_prefix",
    "name_char_len_diff", "name_word_count_diff", "name_comp_exact_match", "name_first_word_match",
    "addr_ratio", "addr_token_sort_ratio", "addr_token_contain", "addr_postal_match", "addr_has_postal",
    "addr_num_exact_match", "is_s2", "is_s3",
    "name_lev_norm", "name_lcs_seq", "name_soundex_match", "addr_lcs_seq",
    "name_tfidf_cosine", "addr_tfidf_cosine", "name_char3_cosine",
    "name_soft_idf_min", "name_soft_idf_mean", "name_idf_wjaccard", "name_max_shared_idf",
    "name_max_unmatched_idf", "addr_idf_wjaccard", "addr_nums_jaccard", "addr_nums_conflict",
    "name_skel_token_contain", "name_skel_ratio", "addr_soft_idf_min", "addr_soft_idf_mean",
    "name_unmatched_count", "name_unmatched_idf_sum", "name_unmatched_min_endrate", "name_acronym_match",
    "grp_name_tfidf_gap", "grp_addr_tfidf_gap", "grp_char3_gap", "grp_soft_gap", "grp_char3_rank",
    "grp_log_ncands", "grp_name_ratio_gap", "grp_soft_min_gap",
]
_F = {n: i for i, n in enumerate(FEATURE_NAMES_V8)}


# --------------------------------------------------------------------- vocab
@dataclass
class CountryVocab:
    name: pl.DataFrame     # tok, nidf, end
    addr: pl.DataFrame     # tok, nidf
    scale: float


def build_country_vocab(prep_targets: pl.DataFrame) -> CountryVocab:
    """Label-free per-country statistics from the target corpus (vectorized)."""
    n = len(prep_targets)
    scale = math.log(1.0 + n)

    def nidf(col):
        return (
            prep_targets.select(pl.col(col).str.split(" ").list.unique().alias("tok")).explode("tok")
            .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
            .group_by("tok").len()
            .with_columns(((1.0 + n / pl.col("len")).log().clip(upper_bound=scale) / scale).cast(pl.Float32).alias("nidf"))
            .select("tok", "nidf")
        )

    toks = prep_targets.select(pl.col("name_clean").str.split(" ").alias("tok")).with_row_index("row")
    toks = toks.with_columns(pl.col("tok").list.len().alias("n")).explode("tok")
    toks = toks.with_columns((pl.col("n") - 1 - pl.int_range(pl.len()).over("row")).alias("from_end"))
    end = (toks.filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
           .group_by("tok").agg((pl.col("from_end") <= 1).mean().cast(pl.Float32).alias("end")))
    return CountryVocab(name=nidf("name_clean").join(end, on="tok", how="left"), addr=nidf("addr_clean"), scale=scale)


# ---------------------------------------------------------------- encoding
def _hash(expr: pl.Expr) -> pl.Expr:
    return (expr.hash(seed=7) // 2).cast(pl.Int64)   # 63-bit, non-negative, never SENT


def _token_table(df: pl.DataFrame, col: str, max_tok: int, vocab: Optional[pl.DataFrame]):
    """Unique tokens per row in first-occurrence order, with counts and vocab stats."""
    t = (df.select(pl.col(col).str.split(" ").alias("tok")).with_row_index("row")
         .explode("tok").with_columns(pl.int_range(pl.len()).over("row").alias("pos"))
         .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
         .group_by("row", "tok").agg(pl.len().alias("tf"), pl.col("pos").min())
         .sort("row", "pos")
         .with_columns(pl.int_range(pl.len()).over("row").alias("slot"))
         .filter(pl.col("slot") < max_tok))
    if vocab is not None:
        t = t.join(vocab, on="tok", how="left")
    return t.with_columns(_hash(pl.col("tok")).alias("id"))


def _dense(n_rows: int, width: int, rows: np.ndarray, slots: np.ndarray, vals: np.ndarray, fill, dtype):
    out = np.full((n_rows, width), fill, dtype=dtype)
    out[rows, slots] = vals
    return out


def _token_shape_features(tokens: List[str]):
    """Per unique token: 126-bit bigram mask (2 words), bigram count, letter mask, length, first char."""
    bg0 = np.zeros(len(tokens), np.int64); bg1 = np.zeros(len(tokens), np.int64)
    nbg = np.zeros(len(tokens), np.float32); letters = np.zeros(len(tokens), np.int64)
    for k, t in enumerate(tokens):
        s = f"^{t}$"
        # zlib.crc32, not hash(): Python's str hash is randomized per process, which made these
        # bigram signatures differ between the training run and the inference run.
        bits = {zlib.crc32(s[i:i + 2].encode("utf-8")) % 126 for i in range(len(s) - 1)}
        nbg[k] = len(bits)
        for b in bits:
            if b < 63:
                bg0[k] |= 1 << b
            else:
                bg1[k] |= 1 << (b - 63)
        m = 0
        for ch in t:
            o = ord(ch) - 97
            m |= (1 << o) if 0 <= o < 26 else (1 << 26)
        letters[k] = m
    lens = np.fromiter((len(t) for t in tokens), np.int16, len(tokens))
    first = np.fromiter((ord(t[0]) if t else 0 for t in tokens), np.int32, len(tokens))
    return bg0, bg1, nbg, letters, lens, first


@dataclass
class RecordBank:
    ids: List[str]
    # name tokens (unique, first-occurrence order) and their stats
    n_id: torch.Tensor; n_tf: torch.Tensor; n_w: torch.Tensor; n_end: torch.Tensor
    n_bg0: torch.Tensor; n_bg1: torch.Tensor; n_nbg: torch.Tensor; n_let: torch.Tensor; n_len: torch.Tensor; n_first: torch.Tensor
    n_norm: torch.Tensor; n_count: torch.Tensor
    a_id: torch.Tensor; a_tf: torch.Tensor; a_w: torch.Tensor
    a_bg0: torch.Tensor; a_bg1: torch.Tensor; a_nbg: torch.Tensor; a_let: torch.Tensor; a_len: torch.Tensor; a_first: torch.Tensor
    a_norm: torch.Tensor; a_count: torch.Tensor
    c3_id: torch.Tensor; c3_w: torch.Tensor
    num_id: torch.Tensor; sk_id: torch.Tensor; init_id: torch.Tensor
    comp8: torch.Tensor; first_word: torch.Tensor; soundex: torch.Tensor; postal: torch.Tensor; house: torch.Tensor
    name_chars: torch.Tensor; n_words: torch.Tensor; is_s2: torch.Tensor; is_s3: torch.Tensor
    # bit-parallel string data (Peq only needed on the S1 / pattern side)
    name_sym: torch.Tensor; name_l: torch.Tensor; nsort_sym: torch.Tensor; nsort_l: torch.Tensor
    addr_sym: torch.Tensor; addr_l: torch.Tensor; asort_sym: torch.Tensor; asort_l: torch.Tensor
    skel_sym: torch.Tensor; skel_l: torch.Tensor
    name_peq: Optional[torch.Tensor] = None; nsort_peq: Optional[torch.Tensor] = None
    addr_peq: Optional[torch.Tensor] = None; asort_peq: Optional[torch.Tensor] = None; skel_peq: Optional[torch.Tensor] = None


def encode_bank(prep: pl.DataFrame, vocab: CountryVocab, id_col: str, pattern_side: bool) -> RecordBank:
    """CPU text encoding (vectorized) -> GPU tensors. pattern_side=True also builds Peq tables."""
    dev = require_cuda()
    n = len(prep)
    df = prep.select(
        pl.col(id_col).alias("id"),
        pl.col("name_clean").fill_null(""), pl.col("addr_clean").fill_null(""),
        pl.col("comp_8").fill_null(""), pl.col("postal_code").fill_null(""), pl.col("house_num").fill_null(""),
        pl.col("name_skel").fill_null(""),
    )
    T = lambda a: torch.as_tensor(np.array(a), device=dev)  # noqa: E731  (copy: Polars arrays are read-only)

    def tok_block(col, width, voc):
        t = _token_table(df, col, width, voc)
        uniq = t.select("tok").unique()
        bg0, bg1, nbg, let, lens, first = _token_shape_features(uniq["tok"].to_list())
        shape = uniq.with_columns(pl.Series("bg0", bg0), pl.Series("bg1", bg1), pl.Series("nbg", nbg),
                                  pl.Series("let", let), pl.Series("tlen", lens), pl.Series("first", first))
        t = t.join(shape, on="tok", how="left")
        r, s = t["row"].to_numpy(), t["slot"].to_numpy()
        w = t["nidf"].fill_null(1.0).to_numpy().astype(np.float32) if voc is not None else np.ones(len(t), np.float32)
        out = {
            "id": _dense(n, width, r, s, t["id"].to_numpy(), SENT, np.int64),
            "tf": _dense(n, width, r, s, t["tf"].to_numpy().astype(np.float32), 0, np.float32),
            "w": _dense(n, width, r, s, w, 0, np.float32),
            "bg0": _dense(n, width, r, s, t["bg0"].to_numpy(), 0, np.int64),
            "bg1": _dense(n, width, r, s, t["bg1"].to_numpy(), 0, np.int64),
            "nbg": _dense(n, width, r, s, t["nbg"].to_numpy(), 0, np.float32),
            "let": _dense(n, width, r, s, t["let"].to_numpy(), 0, np.int64),
            "len": _dense(n, width, r, s, t["tlen"].to_numpy(), 0, np.int16),
            "first": _dense(n, width, r, s, t["first"].to_numpy(), -1, np.int32),
        }
        if voc is not None and "end" in voc.columns:
            out["end"] = _dense(n, width, r, s, t["end"].fill_null(0.0).to_numpy().astype(np.float32), 0, np.float32)
        return {k: T(v) for k, v in out.items()}

    nb = tok_block("name_clean", NT, vocab.name)
    ab = tok_block("addr_clean", AT, vocab.addr)
    n_vec = nb["tf"] * nb["w"]
    a_vec = ab["tf"] * ab["w"]

    # char 3-grams of the space-free padded name, count-weighted and L2-normalized
    padded = df.select((pl.lit(" ") + pl.col("name_clean").str.replace_all(" ", "").str.slice(0, NAME_LEN) + pl.lit(" ")).alias("s"))
    grams = padded.select(pl.concat_list([pl.col("s").str.slice(i, 3) for i in range(NAME_LEN)]).alias("g")).with_row_index("row")
    grams = (grams.explode("g").filter(pl.col("g").str.len_chars() == 3)
             .group_by("row", "g").len().with_columns(_hash(pl.col("g")).alias("id"))
             .sort("row", "len", descending=[False, True])
             .with_columns(pl.int_range(pl.len()).over("row").alias("slot")).filter(pl.col("slot") < NC))
    gl = grams["len"].to_numpy().astype(np.float32)
    c3_id = _dense(n, NC, grams["row"].to_numpy(), grams["slot"].to_numpy(), grams["id"].to_numpy(), SENT, np.int64)
    c3_w = _dense(n, NC, grams["row"].to_numpy(), grams["slot"].to_numpy(), gl, 0, np.float32)
    c3_w = c3_w / np.maximum(np.sqrt((c3_w ** 2).sum(1, keepdims=True)), 1e-9)

    # numeric address tokens (leading zeros stripped), skeleton tokens, initials prefixes
    nums = (df.select(pl.col("addr_clean").str.extract_all(r"\d+[a-z]?").alias("t")).with_row_index("row")
            .explode("t").filter(pl.col("t").is_not_null())
            .with_columns(pl.col("t").str.replace(r"^0+", "")).with_columns(
                pl.when(pl.col("t") == "").then(pl.lit("0")).otherwise(pl.col("t")).alias("t"))
            .unique(["row", "t"], maintain_order=True)
            .with_columns(pl.int_range(pl.len()).over("row").alias("slot")).filter(pl.col("slot") < NN)
            .with_columns(_hash(pl.col("t")).alias("id")))
    num_id = _dense(n, NN, nums["row"].to_numpy(), nums["slot"].to_numpy(), nums["id"].to_numpy(), SENT, np.int64)
    sk = _token_table(df, "name_skel", NK, None)
    sk_id = _dense(n, NK, sk["row"].to_numpy(), sk["slot"].to_numpy(), sk["id"].to_numpy(), SENT, np.int64)
    words = df.select(pl.col("name_clean").str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("ini"))
    ini = words.select([
        pl.when(pl.col("ini").str.len_chars() >= k).then(_hash(pl.col("ini").str.slice(0, k))).otherwise(pl.lit(-1, pl.Int64)).alias(f"k{k}")
        for k in range(2, 2 + NI)
    ]).to_numpy().astype(np.int64)

    # scalar keys
    first_words = df.select(pl.col("name_clean").str.split(" ").list.first().fill_null("")).to_series().to_list()
    soundex = pl.Series([compute_soundex(w) for w in first_words])
    sc = df.select(
        pl.when(pl.col("comp_8") != "").then(_hash(pl.col("comp_8"))).otherwise(-1).alias("comp8"),
        pl.when(pl.col("postal_code") != "").then(_hash(pl.col("postal_code"))).otherwise(-1).alias("postal"),
        pl.when(pl.col("house_num") != "").then(_hash(pl.col("house_num"))).otherwise(-1).alias("house"),
        pl.col("name_clean").str.len_chars().alias("nlen"),
        pl.col("name_clean").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("nw"),
        pl.col("id").str.starts_with("S2").alias("s2"), pl.col("id").str.starts_with("S3").alias("s3"),
    )
    fw = pl.Series(first_words)
    fw_id = pl.select(pl.when(fw != "").then(_hash(pl.lit(fw))).otherwise(-1)).to_series().to_numpy()
    sx_id = pl.select(pl.when(soundex != "").then(_hash(pl.lit(soundex))).otherwise(-1)).to_series().to_numpy()

    # strings for the bit-parallel kernels
    names = df["name_clean"].to_list()
    addrs = df["addr_clean"].to_list()
    nsort = df.select(pl.col("name_clean").str.split(" ").list.sort().list.join(" ")).to_series().to_list()
    asort = df.select(pl.col("addr_clean").str.split(" ").list.sort().list.join(" ")).to_series().to_list()
    skels = df["name_skel"].to_list()
    enc = {k: encode_strings(v, L) for k, v, L in [("name", names, NAME_LEN), ("nsort", nsort, NAME_LEN),
                                                   ("addr", addrs, ADDR_LEN), ("asort", asort, ADDR_LEN),
                                                   ("skel", skels, NAME_LEN)]}

    bank = RecordBank(
        ids=df["id"].to_list(),
        n_id=nb["id"], n_tf=nb["tf"], n_w=nb["w"], n_end=nb["end"], n_bg0=nb["bg0"], n_bg1=nb["bg1"], n_nbg=nb["nbg"],
        n_let=nb["let"], n_len=nb["len"], n_first=nb["first"],
        n_norm=torch.sqrt((n_vec ** 2).sum(1)), n_count=(nb["id"] != SENT).sum(1),
        a_id=ab["id"], a_tf=ab["tf"], a_w=ab["w"], a_bg0=ab["bg0"], a_bg1=ab["bg1"], a_nbg=ab["nbg"],
        a_let=ab["let"], a_len=ab["len"], a_first=ab["first"],
        a_norm=torch.sqrt((a_vec ** 2).sum(1)), a_count=(ab["id"] != SENT).sum(1),
        c3_id=T(c3_id), c3_w=T(c3_w), num_id=T(num_id), sk_id=T(sk_id), init_id=T(ini),
        comp8=T(sc["comp8"].to_numpy()), first_word=T(fw_id), soundex=T(sx_id),
        postal=T(sc["postal"].to_numpy()), house=T(sc["house"].to_numpy()),
        name_chars=T(sc["nlen"].to_numpy().astype(np.float32)), n_words=T(sc["nw"].to_numpy().astype(np.float32)),
        is_s2=T(sc["s2"].to_numpy().astype(np.float32)), is_s3=T(sc["s3"].to_numpy().astype(np.float32)),
        name_sym=T(enc["name"][0]), name_l=T(enc["name"][1]), nsort_sym=T(enc["nsort"][0]), nsort_l=T(enc["nsort"][1]),
        addr_sym=T(enc["addr"][0]), addr_l=T(enc["addr"][1]), asort_sym=T(enc["asort"][0]), asort_l=T(enc["asort"][1]),
        skel_sym=T(enc["skel"][0]), skel_l=T(enc["skel"][1]),
    )
    # sorted copies for set intersections
    for pre in ("n", "a"):
        idt = getattr(bank, f"{pre}_id")
        order = torch.argsort(idt, dim=1)
        keys = _TOK_KEYS + (("end",) if pre == "n" else ())
        for k in keys:
            v = getattr(bank, f"{pre}_{k}")
            setattr(bank, f"{pre}_{k}", torch.gather(v, 1, order))
    for name in ("c3", "num", "sk"):
        idt = getattr(bank, f"{name}_id")
        order = torch.argsort(idt, dim=1)
        setattr(bank, f"{name}_id", torch.gather(idt, 1, order))
        if name == "c3":
            bank.c3_w = torch.gather(bank.c3_w, 1, order)
    if pattern_side:
        bank.name_peq = build_peq(bank.name_sym, 1)
        bank.nsort_peq = build_peq(bank.nsort_sym, 1)
        bank.skel_peq = build_peq(bank.skel_sym, 1)
        bank.addr_peq = build_peq(bank.addr_sym, 3)
        bank.asort_peq = build_peq(bank.asort_sym, 3)
    return bank


# ------------------------------------------------------------ pair features
def _intersect(a_id, b_id):
    """Sorted rows a_id (P, La), b_id (P, Lb): per a-slot match flag and the matched b position."""
    pos = torch.searchsorted(b_id, a_id).clamp(max=b_id.shape[1] - 1)
    hit = (torch.gather(b_id, 1, pos) == a_id) & (a_id != SENT)
    return hit, pos


def _ratio(lcs, la, lb):
    tot = (la + lb).float()
    return torch.where(tot > 0, 2 * lcs.float() / tot.clamp(min=1), torch.ones_like(tot))


def _lcs_norm(lcs, la, lb):
    mx = torch.maximum(la, lb).float()
    return torch.where(mx > 0, lcs.float() / mx.clamp(min=1), torch.ones_like(mx))


def _token_sim(A: dict, B: dict):
    """(P, La, Lb) soft similarity between tokens of A and B."""
    inter = popcount63(A["bg0"][:, :, None] & B["bg0"][:, None, :]) + popcount63(A["bg1"][:, :, None] & B["bg1"][:, None, :])
    dice = 2 * inter.float() / (A["nbg"][:, :, None] + B["nbg"][:, None, :]).clamp(min=1)
    la, lb = A["len"][:, :, None].long(), B["len"][:, None, :].long()
    same_first = A["first"][:, :, None] == B["first"][:, None, :]
    a_in_b = (A["let"][:, :, None] & ~B["let"][:, None, :]) == 0
    b_in_a = (B["let"][:, None, :] & ~A["let"][:, :, None]) == 0
    abbrev = same_first & (((la <= 4) & (la < lb) & a_in_b) | ((lb <= 4) & (lb < la) & b_in_a))
    sim = torch.where(abbrev, torch.clamp(dice, min=ABBREV_SIM), dice)
    exact = A["id"][:, :, None] == B["id"][:, None, :]
    sim = torch.where(exact, torch.ones_like(sim), sim)
    valid = (A["id"][:, :, None] != SENT) & (B["id"][:, None, :] != SENT)
    return torch.where(valid, sim, torch.zeros_like(sim))


def _soft(A: dict, B: dict, sim):
    """IDF-weighted Monge-Elkan both ways + unmatched-token stats."""
    va, vb = A["id"] != SENT, B["id"] != SENT
    best_a = sim.max(2).values                           # (P, La)
    best_b = sim.max(1).values                           # (P, Lb)
    wa, wb = A["w"] * va, B["w"] * vb
    f = (wa * best_a).sum(1) / wa.sum(1).clamp(min=1e-9)
    b = (wb * best_b).sum(1) / wb.sum(1).clamp(min=1e-9)
    un_a = va & (best_a < SOFT_MATCH)
    un_b = vb & (best_b < SOFT_MATCH)
    return f, b, un_a, un_b


def _gather(bank: RecordBank, idx: torch.Tensor, prefix: str, keys) -> dict:
    return {k: getattr(bank, f"{prefix}_{k}")[idx] for k in keys}


_TOK_KEYS = ("id", "tf", "w", "bg0", "bg1", "nbg", "let", "len", "first")


def pair_features_v8(s1: RecordBank, tg: RecordBank, ia: torch.Tensor, ib: torch.Tensor,
                     edge_only: bool = False) -> torch.Tensor:
    """(P, n_pair_features) for pairs (s1 row ia, target row ib). Group features are added separately.

    edge_only=True skips the columns the v10 judge's graph edges never read (name Levenshtein, address
    LCS ratios, acronym match); those columns stay 0 and every other column is unchanged.
    """
    P = ia.shape[0]
    X = torch.zeros(P, len(FEATURE_NAMES_V8), dtype=torch.float32, device=ia.device)

    # --- bit-parallel string ratios (S1 = pattern, target = text)
    nla, nlb = s1.name_l[ia].long(), tg.name_l[ib].long()
    lcs_n = lcs_length(s1.name_peq[ia], s1.name_l[ia], tg.name_sym[ib])
    X[:, _F["name_ratio"]] = _ratio(lcs_n, nla, nlb)
    X[:, _F["name_lcs_seq"]] = _lcs_norm(lcs_n, nla, nlb)
    if not edge_only:
        lev = levenshtein(s1.name_peq[ia], s1.name_l[ia], tg.name_sym[ib], tg.name_l[ib])
        X[:, _F["name_lev_norm"]] = 1 - lev.float() / torch.maximum(nla, nlb).clamp(min=1).float()
    lcs_ns = lcs_length(s1.nsort_peq[ia], s1.nsort_l[ia], tg.nsort_sym[ib])
    X[:, _F["name_token_sort_ratio"]] = _ratio(lcs_ns, s1.nsort_l[ia].long(), tg.nsort_l[ib].long())
    if not edge_only:
        ala, alb = s1.addr_l[ia].long(), tg.addr_l[ib].long()
        lcs_a = lcs_length(s1.addr_peq[ia], s1.addr_l[ia], tg.addr_sym[ib])
        X[:, _F["addr_ratio"]] = _ratio(lcs_a, ala, alb)
        X[:, _F["addr_lcs_seq"]] = _lcs_norm(lcs_a, ala, alb)
        lcs_as = lcs_length(s1.asort_peq[ia], s1.asort_l[ia], tg.asort_sym[ib])
        X[:, _F["addr_token_sort_ratio"]] = _ratio(lcs_as, s1.asort_l[ia].long(), tg.asort_l[ib].long())
    kla, klb = s1.skel_l[ia].long(), tg.skel_l[ib].long()
    lcs_k = lcs_length(s1.skel_peq[ia], s1.skel_l[ia], tg.skel_sym[ib])
    X[:, _F["name_skel_ratio"]] = torch.where((kla > 0) & (klb > 0), _ratio(lcs_k, kla, klb), torch.zeros_like(lcs_k, dtype=torch.float32))

    # common prefix of the two names (up to 8 chars)
    a8, b8 = s1.name_sym[ia, :8], tg.name_sym[ib, :8]
    X[:, _F["name_common_prefix"]] = torch.cumprod((a8 == b8).to(torch.int32), 1).sum(1).float() / 8.0

    # --- scalar comparisons
    eq_known = lambda x, y: ((x == y) & (x != -1)).float()  # noqa: E731
    la, lb = s1.name_chars[ia], tg.name_chars[ib]
    X[:, _F["name_char_len_diff"]] = (la - lb).abs() / torch.maximum(la, lb).clamp(min=1)
    wa, wb = s1.n_words[ia], tg.n_words[ib]
    X[:, _F["name_word_count_diff"]] = (wa - wb).abs() / torch.maximum(wa, wb).clamp(min=1)
    X[:, _F["name_comp_exact_match"]] = eq_known(s1.comp8[ia], tg.comp8[ib])
    X[:, _F["name_first_word_match"]] = eq_known(s1.first_word[ia], tg.first_word[ib])
    X[:, _F["name_soundex_match"]] = eq_known(s1.soundex[ia], tg.soundex[ib])
    pa, pb = s1.postal[ia], tg.postal[ib]
    X[:, _F["addr_postal_match"]] = eq_known(pa, pb)
    X[:, _F["addr_has_postal"]] = ((pa != -1) | (pb != -1)).float()
    X[:, _F["addr_num_exact_match"]] = eq_known(s1.house[ia], tg.house[ib])
    X[:, _F["is_s2"]] = tg.is_s2[ib]
    X[:, _F["is_s3"]] = tg.is_s3[ib]

    # --- name tokens: tf-idf cosine, weighted Jaccard, containment, soft alignment
    A = _gather(s1, ia, "n", _TOK_KEYS + ("end",))
    B = _gather(tg, ib, "n", _TOK_KEYS + ("end",))
    hit, pos = _intersect(A["id"], B["id"])
    va, vb = A["id"] != SENT, B["id"] != SENT
    wA = A["tf"] * A["w"]
    wB = torch.gather(B["tf"] * B["w"], 1, pos)
    dot = (wA * wB * hit).sum(1)
    X[:, _F["name_tfidf_cosine"]] = dot / (s1.n_norm[ia] * tg.n_norm[ib]).clamp(min=1e-9)
    inter_w = (A["w"] * hit).sum(1)
    union_w = (A["w"] * va).sum(1) + (B["w"] * vb).sum(1) - inter_w
    X[:, _F["name_idf_wjaccard"]] = torch.where(union_w > 0, inter_w / union_w.clamp(min=1e-9), torch.zeros_like(union_w))
    X[:, _F["name_max_shared_idf"]] = (A["w"] * hit).max(1).values
    n_inter = hit.sum(1).float()
    n_min = torch.minimum(s1.n_count[ia], tg.n_count[ib]).float()
    X[:, _F["name_token_contain"]] = torch.where(n_min > 0, n_inter / n_min.clamp(min=1), torch.zeros_like(n_min))

    sim = _token_sim(A, B)
    f, b, un_a, un_b = _soft(A, B, sim)
    both = va.any(1) & vb.any(1)
    X[:, _F["name_soft_idf_min"]] = torch.where(both, torch.minimum(f, b), torch.zeros_like(f))
    X[:, _F["name_soft_idf_mean"]] = torch.where(both, 0.5 * (f + b), torch.zeros_like(f))
    X[:, _F["name_max_unmatched_idf"]] = torch.maximum((A["w"] * un_a).max(1).values, (B["w"] * un_b).max(1).values)
    X[:, _F["name_unmatched_count"]] = (un_a.sum(1) + un_b.sum(1)).float()
    X[:, _F["name_unmatched_idf_sum"]] = (A["w"] * un_a).sum(1) + (B["w"] * un_b).sum(1)
    big = torch.full_like(A["end"], 2.0)
    min_end = torch.minimum(torch.where(un_a, A["end"], big).min(1).values, torch.where(un_b, B["end"], big.new_full(B["end"].shape, 2.0)).min(1).values)
    X[:, _F["name_unmatched_min_endrate"]] = torch.where(min_end > 1.5, torch.ones_like(min_end), min_end)

    # acronym: a token of one side (2+ chars) equals the initials prefix of the other side
    if not edge_only:
        ini_a, ini_b = s1.init_id[ia], tg.init_id[ib]
        tok_ok_a = va & (A["len"] >= 2)
        tok_ok_b = vb & (B["len"] >= 2)
        acr = ((A["id"][:, :, None] == ini_b[:, None, :]) & tok_ok_a[:, :, None] & (ini_b[:, None, :] != -1)).any((1, 2))
        acr |= ((B["id"][:, :, None] == ini_a[:, None, :]) & tok_ok_b[:, :, None] & (ini_a[:, None, :] != -1)).any((1, 2))
        X[:, _F["name_acronym_match"]] = acr.float()

    # --- address tokens
    A = _gather(s1, ia, "a", _TOK_KEYS)
    B = _gather(tg, ib, "a", _TOK_KEYS)
    hit, pos = _intersect(A["id"], B["id"])
    va, vb = A["id"] != SENT, B["id"] != SENT
    dot = (A["tf"] * A["w"] * torch.gather(B["tf"] * B["w"], 1, pos) * hit).sum(1)
    X[:, _F["addr_tfidf_cosine"]] = dot / (s1.a_norm[ia] * tg.a_norm[ib]).clamp(min=1e-9)
    inter_w = (A["w"] * hit).sum(1)
    union_w = (A["w"] * va).sum(1) + (B["w"] * vb).sum(1) - inter_w
    X[:, _F["addr_idf_wjaccard"]] = torch.where(union_w > 0, inter_w / union_w.clamp(min=1e-9), torch.zeros_like(union_w))
    a_min = torch.minimum(s1.a_count[ia], tg.a_count[ib]).float()
    X[:, _F["addr_token_contain"]] = torch.where(a_min > 0, hit.sum(1).float() / a_min.clamp(min=1), torch.zeros_like(a_min))
    sim = _token_sim(A, B)
    f, b, _, _ = _soft(A, B, sim)
    both = va.any(1) & vb.any(1)
    X[:, _F["addr_soft_idf_min"]] = torch.where(both, torch.minimum(f, b), torch.zeros_like(f))
    X[:, _F["addr_soft_idf_mean"]] = torch.where(both, 0.5 * (f + b), torch.zeros_like(f))

    # --- char 3-gram cosine, numbers, skeleton tokens
    ca, cb = s1.c3_id[ia], tg.c3_id[ib]
    hit, pos = _intersect(ca, cb)
    X[:, _F["name_char3_cosine"]] = (s1.c3_w[ia] * torch.gather(tg.c3_w[ib], 1, pos) * hit).sum(1)
    na, nb = s1.num_id[ia], tg.num_id[ib]
    hit, _ = _intersect(na, nb)
    ni = hit.sum(1).float()
    nca, ncb = (na != SENT).sum(1).float(), (nb != SENT).sum(1).float()
    has = (nca > 0) & (ncb > 0)
    X[:, _F["addr_nums_jaccard"]] = torch.where(has, ni / (nca + ncb - ni).clamp(min=1), torch.zeros_like(ni))
    X[:, _F["addr_nums_conflict"]] = (has & (ni == 0)).float()
    ka, kb = s1.sk_id[ia], tg.sk_id[ib]
    hit, _ = _intersect(ka, kb)
    kmin = torch.minimum((ka != SENT).sum(1), (kb != SENT).sum(1)).float()
    X[:, _F["name_skel_token_contain"]] = torch.where(kmin > 0, hit.sum(1).float() / kmin.clamp(min=1), torch.zeros_like(kmin))
    return X


def add_group_features(X: torch.Tensor, ia: torch.Tensor) -> None:
    """In place: per-S1 gap to the best candidate, char3 rank, and candidate count."""
    n = X.shape[0]
    if n == 0:
        return
    grp, inv = torch.unique(ia, return_inverse=True)
    G = grp.shape[0]
    for dst, src in [("grp_name_tfidf_gap", "name_tfidf_cosine"), ("grp_addr_tfidf_gap", "addr_tfidf_cosine"),
                     ("grp_char3_gap", "name_char3_cosine"), ("grp_soft_gap", "name_soft_idf_mean"),
                     ("grp_name_ratio_gap", "name_ratio"), ("grp_soft_min_gap", "name_soft_idf_min")]:
        v = X[:, _F[src]]
        gmax = torch.full((G,), -1e9, device=X.device).scatter_reduce(0, inv, v, "amax", include_self=True)
        X[:, _F[dst]] = v - gmax[inv]
    sizes = torch.bincount(inv, minlength=G)
    key = inv.double() * 2.0 - X[:, _F["name_char3_cosine"]].double()
    order = torch.argsort(key)
    starts = torch.cumsum(sizes, 0) - sizes
    ranks = torch.empty(n, device=X.device, dtype=torch.float32)
    ranks[order] = (torch.arange(n, device=X.device) - starts[inv[order]]).float()
    X[:, _F["grp_char3_rank"]] = ranks / (sizes[inv] - 1).clamp(min=1).float()
    X[:, _F["grp_log_ncands"]] = torch.log1p(sizes[inv].float())


def features_v8(s1: RecordBank, tg: RecordBank, ia: torch.Tensor, ib: torch.Tensor, batch: int = 200_000) -> torch.Tensor:
    """Full v8 matrix on the GPU. Pairs must be grouped by S1 (all candidates of an S1 together)."""
    parts = [pair_features_v8(s1, tg, ia[i:i + batch], ib[i:i + batch]) for i in range(0, ia.shape[0], batch)]
    X = torch.cat(parts) if parts else torch.zeros(0, len(FEATURE_NAMES_V8), device=ia.device)
    add_group_features(X, ia)
    return X
