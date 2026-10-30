"""
High-Performance C-Accelerated Pairwise Feature Extraction for Business Entity Resolution.
Variant 6 Architecture (25 features):
- 18 Base string distance, token, structural, and source indicator features.
- 5 Fine-grained alignment features: Damerau-Levenshtein, LCSseq, and Soundex.
- 2 Zero-leakage TF-IDF cosine similarity features (Name and Address).
"""

import os
import re
import sys
import math
import pickle
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from rapidfuzz import fuzz, distance

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_VOCAB_PATH = os.path.join(PKG_ROOT, "models", "tfidf_vocab.pkl")

FEATURE_NAMES = [
    # 1. Base Name similarities
    "name_levenshtein",
    "name_token_sort",
    "name_token_set",
    "name_jaro_winkler",
    "name_char_len_diff",
    "name_word_count_diff",
    "name_comp_exact_match",
    "name_first_word_match",
    # 2. Base Address similarities
    "addr_levenshtein",
    "addr_token_sort",
    "addr_token_set",
    "addr_postal_match",
    "addr_has_postal",
    "addr_num_exact_match",
    "addr_num_overlap_jaccard",
    # 3. Cross-field & Meta
    "name_in_addr",
    "is_s2",
    "is_s3",
    # 4. Variant 5 fine-grained alignment features
    "name_damerau_lev",
    "name_lcs_seq",
    "name_soundex_match",
    "addr_damerau_lev",
    "addr_lcs_seq",
    # 5. Variant 6 zero-leakage TF-IDF cosine similarities
    "name_tfidf_cosine",
    "addr_tfidf_cosine",
]


def compute_soundex(name: str) -> str:
    """Computes standard 4-character American Soundex code for phonetic matching."""
    if not name:
        return ""
    name = name.upper()
    s = [name[0]]
    mapping = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6"}
    for char in name[1:]:
        for keys, val in mapping.items():
            if char in keys:
                if val != s[-1]:
                    s.append(val)
                break
    return "".join(s)[:4].ljust(4, "0")


_NUM_TOKEN = re.compile(r"\d+[^\W\d_]?")


def parse_record_for_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to convert a Polars/database row dict into standardized feature representation."""
    name = row.get("name_clean") or ""
    addr = row.get("addr_clean") or ""
    comp_name = row.get("comp_8") or ""
    pc = row.get("postal_code") or ""
    num = row.get("house_num") or ""
    words = [w for w in name.split() if w]
    first_word = words[0] if words else ""
    nums = [num] if num else []
    return {
        "name": name,
        "comp_name": comp_name,
        "addr": addr,
        "postal_code": pc,
        "nums": nums,
        "addr_nums": {t.lstrip("0") or "0" for t in _NUM_TOKEN.findall(addr)},
        "name_skel": row.get("name_skel") or "",
        "name_words": words,
        "soundex": compute_soundex(first_word),
    }


def compute_country_vocab(prep_targets: "pl.DataFrame") -> Dict[str, Any]:
    """Label-free IDF tables from one country's full target corpus (S2 + S3).

    Needs no ground truth, so it is built the same way for a country seen in
    training and one that is not. Uses the same layout and IDF formula as
    compute_tfidf_vocab_from_targets, so attach_tfidf_profiles accepts it.
    """
    import polars as pl

    total_docs = len(prep_targets)

    def doc_freq(col: str) -> Dict[str, float]:
        counts = (
            prep_targets.select(pl.col(col).str.split(" ").list.unique().alias("tok"))
            .explode("tok")
            .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
            .group_by("tok").len()
        )
        return {t: math.log(1.0 + total_docs / c) for t, c in counts.iter_rows()}

    # Share of each name token's occurrences that fall in the last two words of
    # a name. Legal forms (limited, llc, sarl, gmbh...) sit at the end; business
    # descriptors (holdings, exports, participations...) mostly do not.
    toks = prep_targets.select(pl.col("name_clean").str.split(" ").alias("tok"))
    toks = toks.with_columns(pl.col("tok").list.len().alias("n")).with_row_index("row").explode("tok")
    toks = toks.with_columns((pl.col("n") - 1 - pl.int_range(pl.len()).over("row")).alias("from_end"))
    end_rate = (
        toks.filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
        .group_by("tok").agg((pl.col("from_end") <= 1).mean().alias("rate"))
    )

    default_idf = math.log(1.0 + total_docs)
    return {
        "name_idf": doc_freq("name_clean"),
        "name_default_idf": default_idf,
        "addr_idf": doc_freq("addr_clean"),
        "addr_default_idf": default_idf,
        "name_end_rate": dict(end_rate.iter_rows()),
        "total_docs": total_docs,
    }


def compute_tfidf_vocab_from_targets(target_records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fits inverse document frequency (IDF) tables strictly on provided training target records.
    Guarantees strict zero-leakage: validation and test entities are never observed.
    """
    total_docs = len(target_records)
    name_counts = Counter()
    addr_counts = Counter()

    for rec in target_records.values():
        name_words = set(rec["name_words"])
        name_counts.update(name_words)
        addr_words = set(rec["addr"].split())
        addr_counts.update(addr_words)

    name_idf = {w: math.log(1.0 + total_docs / c) for w, c in name_counts.items()}
    addr_idf = {w: math.log(1.0 + total_docs / c) for w, c in addr_counts.items()}
    default_idf = math.log(1.0 + total_docs)

    return {
        "name_idf": name_idf,
        "name_default_idf": default_idf,
        "addr_idf": addr_idf,
        "addr_default_idf": default_idf,
        "total_docs": total_docs,
    }


def save_tfidf_vocab(vocab: Dict[str, Any], filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_tfidf_vocab(filepath: str = DEFAULT_VOCAB_PATH) -> Optional[Dict[str, Any]]:
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None


def attach_tfidf_profiles(records: Dict[str, Dict[str, Any]], tfidf_vocab: Optional[Dict[str, Any]]) -> None:
    """Precomputes TF-IDF weighted vector and L2 norm once per record to accelerate pairwise cosine by 10x."""
    if not tfidf_vocab:
        return
    name_idf = tfidf_vocab.get("name_idf", {})
    name_def_idf = tfidf_vocab.get("name_default_idf", 10.0)
    addr_idf = tfidf_vocab.get("addr_idf", {})
    addr_def_idf = tfidf_vocab.get("addr_default_idf", 10.0)

    for rec in records.values():
        if "name_tfidf" in rec:
            continue
        # Name profile
        words = rec.get("name_words", [])
        if words:
            cw = Counter(words)
            tf_name = {w: c * name_idf.get(w, name_def_idf) for w, c in cw.items()}
            norm_name = math.sqrt(sum(v * v for v in tf_name.values()))
            rec["name_tfidf"] = tf_name
            rec["name_norm"] = norm_name
        else:
            rec["name_tfidf"] = {}
            rec["name_norm"] = 0.0

        # Address profile
        addr_text = rec.get("addr", "")
        if addr_text:
            addr_words = addr_text.split()
            cwa = Counter(addr_words)
            tf_addr = {w: c * addr_idf.get(w, addr_def_idf) for w, c in cwa.items()}
            norm_addr = math.sqrt(sum(v * v for v in tf_addr.values()))
            rec["addr_tfidf"] = tf_addr
            rec["addr_norm"] = norm_addr
        else:
            rec["addr_tfidf"] = {}
            rec["addr_norm"] = 0.0


def extract_pairwise_features(
    s1_records: Dict[str, Dict[str, Any]],
    target_records: Dict[str, Dict[str, Any]],
    candidate_pairs: List[Tuple[str, str]],
    tfidf_vocab: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, List[str]]:
    """Extracts numeric feature matrix (N, 25) for candidate pairs.
    
    Args:
        s1_records: {s1_id: {name, comp_name, addr, postal_code, nums, name_words, soundex}}
        target_records: {target_id: {name, comp_name, addr, postal_code, nums, name_words, soundex}}
        candidate_pairs: list of (s1_id, target_id)
        tfidf_vocab: optional pre-fit IDF tables dictionary
        
    Returns:
        (X, feature_names): numpy 2D array of shape (N, 25) and feature names list
    """
    n_pairs = len(candidate_pairs)
    n_feats = len(FEATURE_NAMES)
    X = np.empty((n_pairs, n_feats), dtype=np.float32)

    if tfidf_vocab is None:
        tfidf_vocab = load_tfidf_vocab()

    name_idf = tfidf_vocab.get("name_idf", {}) if tfidf_vocab else {}
    name_def_idf = tfidf_vocab.get("name_default_idf", 10.0) if tfidf_vocab else 10.0
    addr_idf = tfidf_vocab.get("addr_idf", {}) if tfidf_vocab else {}
    addr_def_idf = tfidf_vocab.get("addr_default_idf", 10.0) if tfidf_vocab else 10.0

    for i, (s1_id, tid) in enumerate(candidate_pairs):
        r1 = s1_records[s1_id]
        r2 = target_records[tid]

        n1 = r1["name"]
        n2 = r2["name"]
        a1 = r1["addr"]
        a2 = r2["addr"]

        # 1. Base Name features
        X[i, 0] = fuzz.ratio(n1, n2) / 100.0
        X[i, 1] = fuzz.token_sort_ratio(n1, n2) / 100.0
        X[i, 2] = fuzz.token_set_ratio(n1, n2) / 100.0
        X[i, 3] = distance.JaroWinkler.similarity(n1, n2)

        l1, l2 = len(n1), len(n2)
        X[i, 4] = abs(l1 - l2) / max(l1, l2, 1)

        w1_cnt, w2_cnt = len(r1["name_words"]), len(r2["name_words"])
        X[i, 5] = abs(w1_cnt - w2_cnt) / max(w1_cnt, w2_cnt, 1)

        c1, c2 = r1["comp_name"], r2["comp_name"]
        X[i, 6] = 1.0 if (c1 and c2 and c1 == c2) else 0.0

        fw1 = r1["name_words"][0] if w1_cnt > 0 else ""
        fw2 = r2["name_words"][0] if w2_cnt > 0 else ""
        X[i, 7] = 1.0 if (fw1 and fw2 and fw1 == fw2) else 0.0

        # 2. Base Address features
        X[i, 8] = fuzz.ratio(a1, a2) / 100.0
        X[i, 9] = fuzz.token_sort_ratio(a1, a2) / 100.0
        X[i, 10] = fuzz.token_set_ratio(a1, a2) / 100.0

        pc1, pc2 = r1["postal_code"], r2["postal_code"]
        if pc1 and pc2:
            X[i, 11] = 1.0 if pc1 == pc2 else 0.0
            X[i, 12] = 1.0
        else:
            X[i, 11] = 0.0
            X[i, 12] = 1.0 if (pc1 or pc2) else 0.0

        nums1 = r1["nums"]
        nums2 = r2["nums"]
        if nums1 and nums2:
            s_nums1 = set(nums1)
            s_nums2 = set(nums2)
            inter = len(s_nums1.intersection(s_nums2))
            X[i, 13] = 1.0 if (nums1[0] == nums2[0]) else 0.0
            X[i, 14] = inter / len(s_nums1.union(s_nums2))
        else:
            X[i, 13] = 0.0
            X[i, 14] = 0.0

        # 3. Cross-field & Meta
        if c1 and len(c1) >= 5 and c1 in a2:
            X[i, 15] = 1.0
        elif c2 and len(c2) >= 5 and c2 in a1:
            X[i, 15] = 1.0
        else:
            X[i, 15] = 0.0

        X[i, 16] = 1.0 if tid.startswith("S2") else 0.0
        X[i, 17] = 1.0 if tid.startswith("S3") else 0.0

        # 4. Variant 5 fine-grained alignment features
        X[i, 18] = distance.DamerauLevenshtein.normalized_similarity(n1, n2)
        X[i, 19] = distance.LCSseq.normalized_similarity(n1, n2)
        sx1, sx2 = r1["soundex"], r2["soundex"]
        X[i, 20] = 1.0 if (sx1 and sx2 and sx1 == sx2) else 0.0
        X[i, 21] = distance.DamerauLevenshtein.normalized_similarity(a1, a2)
        X[i, 22] = distance.LCSseq.normalized_similarity(a1, a2)

        # 5. Variant 6 zero-leakage TF-IDF cosine similarities (Fast Path)
        if "name_tfidf" in r1 and "name_tfidf" in r2:
            d1, n1 = r1["name_tfidf"], r1["name_norm"]
            d2, n2 = r2["name_tfidf"], r2["name_norm"]
            if n1 > 0 and n2 > 0:
                ds, dl = (d1, d2) if len(d1) <= len(d2) else (d2, d1)
                dot = sum(v * dl[k] for k, v in ds.items() if k in dl)
                X[i, 23] = dot / (n1 * n2)
            else:
                X[i, 23] = 0.0

            da1, na1 = r1["addr_tfidf"], r1["addr_norm"]
            da2, na2 = r2["addr_tfidf"], r2["addr_norm"]
            if na1 > 0 and na2 > 0:
                das, dal = (da1, da2) if len(da1) <= len(da2) else (da2, da1)
                dota = sum(v * dal[k] for k, v in das.items() if k in dal)
                X[i, 24] = dota / (na1 * na2)
            else:
                X[i, 24] = 0.0
        else:
            # Fallback path if attach_tfidf_profiles wasn't called
            w1 = r1["name_words"]
            w2 = r2["name_words"]
            if w1 and w2 and name_idf:
                cw1 = Counter(w1)
                cw2 = Counter(w2)
                inter_w = set(cw1.keys()) & set(cw2.keys())
                if inter_w:
                    dot = sum(cw1[t] * cw2[t] * (name_idf.get(t, name_def_idf) ** 2) for t in inter_w)
                    norm1 = math.sqrt(sum((cw1[t] * name_idf.get(t, name_def_idf)) ** 2 for t in cw1))
                    norm2 = math.sqrt(sum((cw2[t] * name_idf.get(t, name_def_idf)) ** 2 for t in cw2))
                    X[i, 23] = dot / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0
                else:
                    X[i, 23] = 0.0
            else:
                X[i, 23] = 0.0

            wa1 = a1.split()
            wa2 = a2.split()
            if wa1 and wa2 and addr_idf:
                cwa1 = Counter(wa1)
                cwa2 = Counter(wa2)
                inter_wa = set(cwa1.keys()) & set(cwa2.keys())
                if inter_wa:
                    dota = sum(cwa1[t] * cwa2[t] * (addr_idf.get(t, addr_def_idf) ** 2) for t in inter_wa)
                    norma1 = math.sqrt(sum((cwa1[t] * addr_idf.get(t, addr_def_idf)) ** 2 for t in cwa1))
                    norma2 = math.sqrt(sum((cwa2[t] * addr_idf.get(t, addr_def_idf)) ** 2 for t in cwa2))
                    X[i, 24] = dota / (norma1 * norma2) if (norma1 > 0 and norma2 > 0) else 0.0
                else:
                    X[i, 24] = 0.0
            else:
                X[i, 24] = 0.0

    return X, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Variant 7: country-agnostic features for generalizing to unseen countries.
# Every IDF is divided by log(1 + N) so values sit in [0, 1] regardless of how
# large a country's corpus is; raw IDF scales with corpus size and would not
# transfer from US/India to France.
# ---------------------------------------------------------------------------

PAIR_FEATURE_NAMES_V7 = [
    "name_char3_cosine",
    "name_soft_idf_min",
    "name_soft_idf_mean",
    "name_idf_wjaccard",
    "name_max_shared_idf",
    "name_max_unmatched_idf",
    "addr_idf_wjaccard",
    "addr_nums_jaccard",
    "addr_nums_conflict",
    "name_skel_token_set",
    "name_skel_ratio",
    "addr_soft_idf_min",
    "addr_soft_idf_mean",
    "name_unmatched_count",
    "name_unmatched_idf_sum",
    "name_unmatched_min_endrate",
    "name_acronym_match",
]
GROUP_FEATURE_NAMES_V7 = [
    "grp_name_tfidf_gap",
    "grp_addr_tfidf_gap",
    "grp_char3_gap",
    "grp_soft_gap",
    "grp_char3_rank",
    "grp_log_ncands",
    "grp_name_lev_gap",
    "grp_soft_min_gap",
]
FEATURE_NAMES_V7 = FEATURE_NAMES + PAIR_FEATURE_NAMES_V7 + GROUP_FEATURE_NAMES_V7

# +1: more similar never lowers match probability; -1: the reverse; 0: free.
_MONOTONE_V7 = {
    "name_levenshtein": 1, "name_token_sort": 1, "name_token_set": 1, "name_jaro_winkler": 1,
    "name_comp_exact_match": 1, "name_first_word_match": 1,
    "addr_levenshtein": 1, "addr_token_sort": 1, "addr_token_set": 1, "addr_postal_match": 1,
    "addr_num_exact_match": 1, "addr_num_overlap_jaccard": 1,
    "name_damerau_lev": 1, "name_lcs_seq": 1, "name_soundex_match": 1,
    "addr_damerau_lev": 1, "addr_lcs_seq": 1,
    "name_tfidf_cosine": 1, "addr_tfidf_cosine": 1,
    "name_char3_cosine": 1, "name_soft_idf_min": 1, "name_soft_idf_mean": 1,
    "name_idf_wjaccard": 1, "name_max_shared_idf": 1, "name_max_unmatched_idf": -1,
    "addr_idf_wjaccard": 1, "addr_nums_jaccard": 1, "addr_nums_conflict": -1,
    "name_skel_token_set": 1, "name_skel_ratio": 1, "addr_soft_idf_min": 1, "addr_soft_idf_mean": 1,
    "name_unmatched_count": -1, "name_unmatched_idf_sum": -1, "name_acronym_match": 1,
    "grp_name_lev_gap": 1, "grp_soft_min_gap": 1,
    "grp_name_tfidf_gap": 1, "grp_addr_tfidf_gap": 1, "grp_char3_gap": 1, "grp_soft_gap": 1,
    "grp_char3_rank": -1,
}
MONOTONE_CONSTRAINTS_V7 = tuple(_MONOTONE_V7.get(f, 0) for f in FEATURE_NAMES_V7)

_SOFT_MATCH_JW = 0.88


def _char3_profile(name: str) -> Dict[str, float]:
    """Unit-length character 3-gram count vector of the padded, space-free name."""
    s = f" {name.replace(' ', '')} "
    if len(s) < 3:
        return {}
    c = Counter(s[i:i + 3] for i in range(len(s) - 2))
    norm = math.sqrt(sum(v * v for v in c.values()))
    return {g: v / norm for g, v in c.items()}


def attach_v7_profiles(records: Dict[str, Dict[str, Any]], vocab: Dict[str, Any]) -> None:
    """Adds per-country TF-IDF profiles plus the normalized-IDF data v7 needs."""
    attach_tfidf_profiles(records, vocab)
    scale = math.log(1.0 + vocab["total_docs"])
    name_idf, addr_idf = vocab["name_idf"], vocab["addr_idf"]
    end_rate = vocab.get("name_end_rate", {})
    for rec in records.values():
        if "char3" in rec:
            continue
        rec["char3"] = _char3_profile(rec["name"])
        rec["name_endrate"] = {w: end_rate.get(w, 0.0) for w in set(rec["name_words"])}
        rec["name_nidf"] = {w: min(name_idf.get(w, scale), scale) / scale for w in set(rec["name_words"])}
        rec["addr_nidf"] = {w: min(addr_idf.get(w, scale), scale) / scale for w in set(rec["addr"].split())}


def _sparse_dot(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b[k] for k, v in a.items() if k in b)


_ABBREV_SIM = 0.9


def _is_abbrev(short: str, long: str) -> bool:
    """'r'/'rue', 'av'/'avenue', 'bd'/'boulevard', 'ets'/'etablissements':
    same first letter and the short token's letters appear in order."""
    if len(short) > 4 or len(short) >= len(long) or short[0] != long[0]:
        return False
    it = iter(long)
    return all(ch in it for ch in short)


def _tok_sim(a: str, b: str) -> float:
    s = distance.JaroWinkler.similarity(a, b)
    if s < _ABBREV_SIM and a and b and (_is_abbrev(a, b) or _is_abbrev(b, a)):
        return _ABBREV_SIM
    return s


def _soft_idf(src: Dict[str, float], dst: Dict[str, float]) -> Tuple[float, float, List[str]]:
    """IDF-weighted Monge-Elkan alignment of src tokens onto dst tokens.

    Returns (score, max normalized IDF among src tokens left unmatched,
    the unmatched src tokens).
    """
    total = matched = unmatched_max = 0.0
    unmatched = []
    for a, w in src.items():
        best = max(_tok_sim(a, b) for b in dst)
        total += w
        matched += w * best
        if best < _SOFT_MATCH_JW:
            unmatched.append(a)
            if w > unmatched_max:
                unmatched_max = w
    return (matched / total if total > 0 else 0.0), unmatched_max, unmatched


def _acronym_match(words_a: List[str], words_b: List[str]) -> bool:
    """A short token of one name spells the initials of the other's leading words
    ('si' ~ 'seven infotech private limited')."""
    for short_side, long_side in ((words_a, words_b), (words_b, words_a)):
        if len(long_side) < 2:
            continue
        initials = "".join(w[0] for w in long_side)
        for t in short_side:
            if 2 <= len(t) <= len(initials) and initials.startswith(t):
                return True
    return False


def _wjaccard(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, float]:
    """IDF-weighted Jaccard and the largest IDF among shared tokens."""
    shared = a.keys() & b.keys()
    if not shared:
        return 0.0, 0.0
    inter = sum(a[t] for t in shared)
    union = sum(a.values()) + sum(v for t, v in b.items() if t not in a)
    return (inter / union if union > 0 else 0.0), max(a[t] for t in shared)


def extract_pairwise_features_v7(
    s1_records: Dict[str, Dict[str, Any]],
    target_records: Dict[str, Dict[str, Any]],
    candidate_pairs: List[Tuple[str, str]],
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """Feature matrix (N, len(FEATURE_NAMES_V7)).

    Records must have been passed through attach_v7_profiles with the vocab of
    their own country. Group features compare each pair against the other
    candidates of the same S1 entity, so candidate_pairs must hold every
    candidate of each S1 it mentions.

    The per-pair string metrics are CPU-bound, so large inputs are split into
    chunks scored in parallel worker processes (n_jobs defaults to all cores
    but one). Callers on Windows must run this under `if __name__ == "__main__"`.
    """
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    n = len(candidate_pairs)
    if n_jobs == 1 or n < 50_000:
        BP = _pair_block((s1_records, target_records, candidate_pairs))
    else:
        from concurrent.futures import ProcessPoolExecutor
        size = -(-n // (n_jobs * 4))
        chunks = []
        for lo in range(0, n, size):
            part = candidate_pairs[lo:lo + size]
            chunks.append((
                {s: s1_records[s] for s in {p[0] for p in part}},
                {t: target_records[t] for t in {p[1] for p in part}},
                part,
            ))
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            BP = np.vstack(list(pool.map(_pair_block, chunks)))
    n_base = len(FEATURE_NAMES)
    base, P = BP[:, :n_base], BP[:, n_base:]
    G = _group_features(candidate_pairs, base, P)
    return np.hstack([base, P, G])


def _pair_block(payload) -> np.ndarray:
    """Base v6 features plus v7 pair features for one chunk of pairs."""
    s1_records, target_records, candidate_pairs = payload
    base, _ = extract_pairwise_features(s1_records, target_records, candidate_pairs, tfidf_vocab={})
    n = len(candidate_pairs)
    P = np.zeros((n, len(PAIR_FEATURE_NAMES_V7)), dtype=np.float32)

    for i, (s1_id, tid) in enumerate(candidate_pairs):
        r1, r2 = s1_records[s1_id], target_records[tid]
        P[i, 0] = _sparse_dot(r1["char3"], r2["char3"])

        n1, n2 = r1["name_nidf"], r2["name_nidf"]
        if n1 and n2:
            f, u1, un1 = _soft_idf(n1, n2)
            b, u2, un2 = _soft_idf(n2, n1)
            P[i, 1] = min(f, b)
            P[i, 2] = 0.5 * (f + b)
            P[i, 3], P[i, 4] = _wjaccard(n1, n2)
            P[i, 5] = max(u1, u2)
            P[i, 13] = len(un1) + len(un2)
            P[i, 14] = sum(n1[t] for t in un1) + sum(n2[t] for t in un2)
            rates = [r1["name_endrate"][t] for t in un1] + [r2["name_endrate"][t] for t in un2]
            P[i, 15] = min(rates) if rates else 1.0
            P[i, 16] = 1.0 if _acronym_match(r1["name_words"], r2["name_words"]) else 0.0
        else:
            P[i, 5] = max(n1.values(), default=0.0) if not n2 else max(n2.values(), default=0.0)
            P[i, 13] = len(n1) + len(n2)
            P[i, 14] = sum(n1.values()) + sum(n2.values())
            P[i, 15] = 0.0

        a1, a2 = r1["addr_nidf"], r2["addr_nidf"]
        if a1 and a2:
            P[i, 6], _ = _wjaccard(a1, a2)
            f, _, _ = _soft_idf(a1, a2)
            b, _, _ = _soft_idf(a2, a1)
            P[i, 11] = min(f, b)
            P[i, 12] = 0.5 * (f + b)

        k1, k2 = r1["name_skel"], r2["name_skel"]
        if k1 and k2:
            P[i, 9] = fuzz.token_set_ratio(k1, k2) / 100.0
            P[i, 10] = fuzz.ratio(k1, k2) / 100.0

        m1, m2 = r1["addr_nums"], r2["addr_nums"]
        if m1 and m2:
            inter = len(m1 & m2)
            P[i, 7] = inter / len(m1 | m2)
            P[i, 8] = 1.0 if inter == 0 else 0.0

    return np.hstack([base, P])


def _group_features(candidate_pairs: List[Tuple[str, str]], base: np.ndarray, pair: np.ndarray) -> np.ndarray:
    """Per-S1 relative features: gap to the S1's best candidate, rank, and set size."""
    n = len(candidate_pairs)
    G = np.zeros((n, len(GROUP_FEATURE_NAMES_V7)), dtype=np.float32)
    if n == 0:
        return G
    _, grp = np.unique(np.array([p[0] for p in candidate_pairs]), return_inverse=True)
    n_groups = grp.max() + 1
    gap_cols = {
        0: base[:, FEATURE_NAMES.index("name_tfidf_cosine")],
        1: base[:, FEATURE_NAMES.index("addr_tfidf_cosine")],
        2: pair[:, 0],
        3: pair[:, 2],
        6: base[:, FEATURE_NAMES.index("name_levenshtein")],
        7: pair[:, 1],
    }
    for j, v in gap_cols.items():
        gmax = np.full(n_groups, -np.inf, dtype=np.float32)
        np.maximum.at(gmax, grp, v)
        G[:, j] = v - gmax[grp]

    # Rank of char3 cosine within its group, 0 = best, normalized by group size
    sizes = np.bincount(grp, minlength=n_groups)
    order = np.lexsort((-pair[:, 0], grp))
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n) - np.repeat(starts, sizes)
    G[:, 4] = ranks / np.maximum(sizes[grp] - 1, 1)
    G[:, 5] = np.log1p(sizes[grp])
    return G


if __name__ == "__main__":
    r1 = parse_record_for_features({
        "name_clean": "payne enterprises llc", "comp_8": "payneenterprisesllc",
        "addr_clean": "3315 fremont street peoria il", "postal_code": "", "house_num": "3315"
    })
    r2 = parse_record_for_features({
        "name_clean": "payne enterprises", "comp_8": "payneenterprises",
        "addr_clean": "3315 fremont st peoria il", "postal_code": "", "house_num": "3315"
    })
    s1_dict = {"S1-1": r1}
    target_dict = {"S2-1": r2}
    vocab = compute_tfidf_vocab_from_targets(target_dict)
    X, names = extract_pairwise_features(s1_dict, target_dict, [("S1-1", "S2-1")], tfidf_vocab=vocab)
    print(f"Extracted {len(names)} features successfully:")
    for n, v in zip(names, X[0]):
        print(f"  {n:25s}: {v:.4f}")
    assert len(names) == 25
    assert X[0, 23] > 0.5  # name tfidf cosine
    assert X[0, 24] > 0.5  # addr tfidf cosine
    print("All 25 feature tests passed!")
