"""
Scalable Multi-Path Blocking Engine for Business Entity Resolution.
Features:
- Country-partitioned execution (US, India, France) to keep memory lightweight.
- Frequency capping: suppresses generic/hub keys with > 60 occurrences.
- High-recall multi-path keys:
    1. Compressed name prefix (8 chars)
    2. First 2 salient words
    3. Rare first word (len >= 5, target count <= 60)
    4. Address anchor: (postal code + house number)
    5. Locality anchor: (street number + salient address token)
"""

import os
import sys
import re
import time
from collections import Counter, defaultdict
from typing import Dict, Set, List, Tuple, Any
import polars as pl

from translit import indic_to_latin

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8")

# Common generic stop words to ignore in single-token blocking
STOP_WORDS = {
    "the", "and", "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "co", "company", "llc", "llp", "pvt", "private", "services", "enterprises",
    "solutions", "group", "holdings", "management", "international", "associates",
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "near", "opp",
    "floor", "fl", "unit", "apt", "plot", "sector", "building", "bldg", "shop", "no",
    "hotel", "restaurant", "store", "center", "centre", "dr", "mr", "ms", "de", "du", "la"
}

# Indian geographic stops to avoid generic hubs in address bigrams
INDIAN_GEO_STOPS = {
    "india", "karnataka", "maharashtra", "delhi", "gujarat", "tamil", "nadu",
    "uttar", "pradesh", "bengal", "west", "haryana", "rajasthan", "telangana",
    "andhra", "kerala", "punjab", "bihar", "odisha", "mumbai", "delhi", "bangalore",
    "bengaluru", "chennai", "hyderabad", "kolkata", "pune", "ahmedabad", "jaipur"
}
ALL_STOPS = STOP_WORDS.union(INDIAN_GEO_STOPS)

MAX_CANDIDATES_PER_KEY = 50
MAX_CANDIDATES_PER_S1 = 120
MIN_NAME_TOKEN_LEN = 3   # short rare words are kept; hub pruning drops common ones
MAX_NAME_TOKENS = 4
MAX_REFINE_HUB = 5000     # name keys more common than this are too generic to refine
REFINE_TOKENS = 6


def refine_tokens(addr: str) -> List[str]:
    """Distinctive address tokens (numbers and words of 3+ letters, minus generic
    geography/street words) used to split over-common name keys."""
    out: List[str] = []
    for w in (addr or "").split():
        if (w[0].isdigit() or len(w) >= 3) and w not in ALL_STOPS and w not in out:
            out.append(w)
            if len(out) == REFINE_TOKENS:
                break
    return out


def _transliterate(expr: pl.Expr) -> pl.Expr:
    return expr.map_batches(
        lambda s: pl.Series([indic_to_latin(v) if v else v for v in s.to_list()], dtype=pl.Utf8),
        return_dtype=pl.Utf8,
    )


def _clean_text(expr: pl.Expr) -> pl.Expr:
    """Lowercases and strips punctuation, keeping every script comparable.

    Indic scripts are transliterated to Latin so e.g. Hindi-script names meet their
    English spellings; Latin diacritics are folded (societe == société);
    dotted acronyms are joined (S.A.S. == SAS); whitespace is collapsed so
    splitting on " " never yields empty tokens.
    """
    acronym = expr
    for n in (4, 3, 2):
        pat = r"\b" + r"\.".join([r"(\p{L})"] * n) + r"\b\.?"
        acronym = acronym.str.replace_all(pat, "".join(f"${{{i}}}" for i in range(1, n + 1)))
    return (
        _transliterate(acronym)
        .str.normalize("NFKD")
        .str.replace_all(r"(\p{Latin})\p{Mn}+", "${1}")
        .str.to_lowercase()
        .str.replace_all(r"[^\p{L}\p{N}\p{M}\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


_SKELETON_RULES = [
    (r"chh", "ch"), (r"ph", "f"), (r"bh", "b"), (r"kh", "k"), (r"gh", "g"), (r"jh", "j"),
    (r"th", "t"), (r"dh", "d"), (r"sh", "s"), (r"ck", "k"), (r"q", "k"), (r"x", "ks"),
    (r"w", "v"), (r"z", "s"), (r"c([eiy])", "s${1}"), (r"c", "k"),
    (r"\B[aeiouy]+", ""),
] + [(c + c, c) for c in "bdfgjklmnprstv"]


def phonetic_skeleton(expr: pl.Expr) -> pl.Expr:
    """Consonant skeleton of each word: 'modern finance' and the transliterated
    'modarn phaainens' both become 'mdrn fns'. Language-agnostic sound-alike key."""
    for pat, rep in _SKELETON_RULES:
        expr = expr.str.replace_all(pat, rep)
    return expr


def extract_record_features(df: pl.DataFrame, is_s1: bool = False) -> pl.DataFrame:
    """Vectorized feature extraction via Polars."""
    id_col = "source1_entity_id" if "source1_entity_id" in df.columns else "entity_id"
    out_id = "s1_id" if is_s1 else "target_id"

    return df.select([
        pl.col(id_col).alias(out_id),
        pl.col("country"),
        pl.col("business_name").fill_null("").alias("name_raw"),
        pl.col("business_address").fill_null("").alias("addr_raw"),
    ]).with_columns([
        _clean_text(pl.col("name_raw").str.replace_all(r"\[.*?\]|\(.*?\)", " ")).alias("name_clean"),
        _clean_text(pl.col("addr_raw")).alias("addr_clean"),
    ]).with_columns([
        # Key 1: Compressed name prefix (8 chars)
        pl.col("name_clean").str.replace_all(r"\s+", "").str.slice(0, 8).alias("comp_8"),
        # Key 2: First 2 words joined
        pl.col("name_clean").str.split(" ").list.slice(0, 2).list.join("_").alias("words_2"),
        # Key 3: First word
        pl.col("name_clean").str.split(" ").list.first().alias("first_word"),
        # Key 4: Postal code
        pl.col("addr_raw").str.extract(r"\b(\d{5,6})\b", 1).alias("postal_code"),
        # Key 5: House/plot number (stripped of leading zeros)
        pl.col("addr_clean").str.extract(r"\b0*(\d+[a-z]?)\b", 1).alias("house_num"),
        phonetic_skeleton(pl.col("name_clean")).alias("name_skel"),
    ]).with_columns([
        # Key 6: compressed phonetic skeleton prefix (sound-alike / cross-script names)
        pl.col("name_skel").str.replace_all(" ", "").str.slice(0, 6).alias("skel_6"),
    ])


class MultiPathBlocker:
    def __init__(self, max_per_key: int = MAX_CANDIDATES_PER_KEY, max_per_s1: int = MAX_CANDIDATES_PER_S1):
        self.max_per_key = max_per_key
        self.max_per_s1 = max_per_s1
        self.indexes: Dict[str, Dict[str, Any]] = {}

    def build_index_for_country(self, df_targets: pl.DataFrame):
        """Builds multi-path inverted index for target records (S2 + S3) of one country."""
        if df_targets.is_empty():
            return
        t0 = time.time()
        country = df_targets["country"][0]
        print(f"[{country}] Building index over {len(df_targets):,} target records...")

        # Indexes for this country
        idx_comp8 = defaultdict(list)
        idx_comp5 = defaultdict(list)
        idx_words2 = defaultdict(list)
        idx_name_tok = defaultdict(list)
        idx_pc_num = defaultdict(list)
        idx_num_tok = defaultdict(list)
        idx_addr_bi = defaultdict(list)
        idx_skel6 = defaultdict(list)

        # Convert to dictionary/arrays for blazing fast Python iteration
        rows = df_targets.select(["target_id", "comp_8", "words_2", "name_clean", "postal_code", "house_num", "addr_clean", "skel_6"]).to_dicts()

        for r in rows:
            tid = r["target_id"]
            c8 = r["comp_8"]
            w2 = r["words_2"]
            nc = r["name_clean"]
            pc = r["postal_code"]
            num = r["house_num"]
            ac = r["addr_clean"]
            sk6 = r["skel_6"]

            if c8 and len(c8) >= 5:
                idx_comp8[c8].append(tid)
                idx_comp5[c8[:5]].append(tid)

            if w2 and "_" in w2 and len(w2) >= 5:
                idx_words2[w2].append(tid)

            # Key 3: Salient name words (catches word-order swaps)
            if nc:
                nwords = [w for w in nc.split() if len(w) >= MIN_NAME_TOKEN_LEN and w not in STOP_WORDS]
                for nw in nwords[:MAX_NAME_TOKENS]:
                    idx_name_tok[nw].append(tid)

            if sk6 and len(sk6) >= 4:
                idx_skel6[sk6].append(tid)

            if pc and num:
                idx_pc_num[(pc, num)].append(tid)

            # Key 5: Street number + salient address word
            if num and ac:
                addr_words = [w for w in ac.split() if len(w) >= 4 and w not in STOP_WORDS]
                for aw in addr_words[:2]:
                    idx_num_tok[(num, aw)].append(tid)

            # Address bigrams (first used for India, where misses lack postal codes)
            if ac:
                words = [w for w in ac.split() if len(w) >= 5 and w not in ALL_STOPS]
                for i in range(min(len(words) - 1, 3)):
                    pair = tuple(sorted([words[i], words[i+1]]))
                    idx_addr_bi[pair].append(tid)

        # Apply frequency capping: truncate to cap, drop extreme hubs (> 150)
        def prune_index(idx, cap, max_hub=150):
            pruned = {}
            for k, v in idx.items():
                if len(v) <= max_hub:
                    pruned[k] = v[:cap]
            return pruned

        # Name keys shared by more than max_per_key targets are not truncated in file
        # order (that dropped true matches of common names at random). They are
        # refined instead: (name key, address token) sub-keys, each kept when small.
        addr_of = {r["target_id"]: r["addr_clean"] for r in rows}
        refined: Dict[Any, List[str]] = defaultdict(list)
        hub_keys: Dict[str, Set[Any]] = {}
        name_indexes = {"comp8": idx_comp8, "comp5": idx_comp5, "words2": idx_words2,
                        "name_tok": idx_name_tok, "skel6": idx_skel6}
        pruned_name = {}
        for iname, index in name_indexes.items():
            small, hubs = {}, set()
            for k, v in index.items():
                if len(v) <= self.max_per_key:
                    small[k] = v
                elif len(v) <= MAX_REFINE_HUB:
                    hubs.add(k)
                    for tid in v:
                        for tok in refine_tokens(addr_of[tid]):
                            refined[(iname, k, tok)].append(tid)
            pruned_name[iname] = small
            hub_keys[iname] = hubs
        refined = {k: v for k, v in refined.items() if len(v) <= self.max_per_key}
        del addr_of

        pruned_comp8 = pruned_name["comp8"]
        pruned_comp5 = pruned_name["comp5"]
        pruned_words2 = pruned_name["words2"]
        pruned_name_tok = pruned_name["name_tok"]
        pruned_skel6 = pruned_name["skel6"]
        pruned_pc_num = prune_index(idx_pc_num, self.max_per_key)
        pruned_num_tok = prune_index(idx_num_tok, self.max_per_key)
        pruned_addr_bi = prune_index(idx_addr_bi, 30, max_hub=100)

        self.indexes[country] = {
            "refined": refined,
            "hubs": hub_keys,
            "comp8": pruned_comp8,
            "comp5": pruned_comp5,
            "words2": pruned_words2,
            "name_tok": pruned_name_tok,
            "pc_num": pruned_pc_num,
            "num_tok": pruned_num_tok,
            "addr_bi": pruned_addr_bi,
            "skel6": pruned_skel6,
        }
        print(f"[{country}] Index built in {time.time() - t0:.2f}s (Comp8: {len(pruned_comp8):,}, Comp5: {len(pruned_comp5):,}, Words2: {len(pruned_words2):,}, NameTok: {len(pruned_name_tok):,}, Num+Tok: {len(pruned_num_tok):,}, AddrBi: {len(pruned_addr_bi):,}, Skel6: {len(pruned_skel6):,}, Refined: {len(refined):,}).")

    def query_s1_for_country(self, df_s1: pl.DataFrame) -> Dict[str, Set[str]]:
        """Queries candidates for S1 entities of one country."""
        if df_s1.is_empty():
            return {}
        country = df_s1["country"][0]
        if country not in self.indexes:
            return {row["s1_id"]: set() for row in df_s1.iter_rows(named=True)}

        idx = self.indexes[country]
        idx_comp8 = idx["comp8"]
        idx_comp5 = idx["comp5"]
        idx_words2 = idx["words2"]
        idx_name_tok = idx["name_tok"]
        idx_pc_num = idx["pc_num"]
        idx_num_tok = idx["num_tok"]
        idx_addr_bi = idx.get("addr_bi", {})
        idx_skel6 = idx.get("skel6", {})
        refined = idx.get("refined", {})
        hubs = idx.get("hubs", {})

        t0 = time.time()
        print(f"[{country}] Querying {len(df_s1):,} S1 entities...")

        rows = df_s1.select(["s1_id", "comp_8", "words_2", "name_clean", "postal_code", "house_num", "addr_clean", "skel_6"]).to_dicts()
        candidates: Dict[str, Set[str]] = {}

        for r in rows:
            sid = r["s1_id"]
            c8 = r["comp_8"]
            w2 = r["words_2"]
            nc = r["name_clean"]
            pc = r["postal_code"]
            num = r["house_num"]
            ac = r["addr_clean"]
            sk6 = r["skel_6"]

            # Count how many blocking keys each target shares with this S1 record
            hits: Counter = Counter()
            if c8 and c8 in idx_comp8:
                hits.update(idx_comp8[c8])
            if c8 and len(c8) >= 5 and c8[:5] in idx_comp5:
                hits.update(idx_comp5[c8[:5]])
            if w2 and w2 in idx_words2:
                hits.update(idx_words2[w2])
            if nc:
                nwords = [w for w in nc.split() if len(w) >= MIN_NAME_TOKEN_LEN and w not in STOP_WORDS]
                for nw in nwords[:MAX_NAME_TOKENS]:
                    if nw in idx_name_tok:
                        hits.update(idx_name_tok[nw])
            if sk6 and sk6 in idx_skel6:
                hits.update(idx_skel6[sk6])
            if pc and num and (pc, num) in idx_pc_num:
                hits.update(idx_pc_num[(pc, num)])
            if num and ac:
                addr_words = [w for w in ac.split() if len(w) >= 4 and w not in STOP_WORDS]
                for aw in addr_words[:2]:
                    if (num, aw) in idx_num_tok:
                        hits.update(idx_num_tok[(num, aw)])

            # Over-common name keys: look up (key, address token) refinements
            if refined:
                s_name_keys = [("comp8", c8), ("comp5", c8[:5] if c8 and len(c8) >= 5 else None),
                               ("words2", w2), ("skel6", sk6)]
                if nc:
                    s_name_keys += [("name_tok", w) for w in nc.split()
                                    if len(w) >= MIN_NAME_TOKEN_LEN and w not in STOP_WORDS][:MAX_NAME_TOKENS]
                s_tokens = None
                for iname, key in s_name_keys:
                    if key and key in hubs.get(iname, ()):
                        if s_tokens is None:
                            s_tokens = refine_tokens(ac)
                        for tok in s_tokens:
                            hit = refined.get((iname, key, tok))
                            if hit:
                                hits.update(hit)

            # Address bigram query
            if ac and idx_addr_bi:
                words = [w for w in ac.split() if len(w) >= 5 and w not in ALL_STOPS]
                for i in range(min(len(words) - 1, 3)):
                    pair = tuple(sorted([words[i], words[i+1]]))
                    if pair in idx_addr_bi:
                        hits.update(idx_addr_bi[pair])

            # Over the cap, keep the targets sharing the most keys (ties broken by
            # ID for determinism). Sorting by ID alone kept every S2 before any S3.
            if len(hits) > self.max_per_s1:
                ranked = sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))
                cands = {tid for tid, _ in ranked[:self.max_per_s1]}
            else:
                cands = set(hits)

            candidates[sid] = cands

        print(f"[{country}] Querying completed in {time.time() - t0:.2f}s.")
        return candidates


def run_benchmark():
    DATA_DIR = r"c:\Users\realr\OneDrive\Desktop\Amazon\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
    print("Loading Ground Truth...")
    df_gt = pl.read_csv(os.path.join(DATA_DIR, "train_ground_truth.tsv"), separator="\t").head(40000)
    sample_s1_ids = set(df_gt["source1_entity_id"].to_list())

    gt_map = {}
    for row in df_gt.iter_rows(named=True):
        sid = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        gt_map[sid] = set(x.strip() for x in matched.split(",") if x.strip()) if matched else set()

    total_true = sum(len(v) for v in gt_map.values())
    print(f"Evaluating on {len(df_gt):,} S1 entities ({total_true:,} true matches).")

    # Load S1
    print("Loading S1...")
    df_s1 = pl.read_csv(os.path.join(DATA_DIR, "train_source1.tsv"), separator="\t")
    df_s1 = df_s1.filter(pl.col("entity_id").is_in(list(sample_s1_ids)))
    prep_s1 = extract_record_features(df_s1, is_s1=True)

    # Load S2 & S3
    print("Loading S2 and S3...")
    df_s2 = pl.read_csv(os.path.join(DATA_DIR, "train_source2.tsv"), separator="\t")
    df_s3 = pl.read_csv(os.path.join(DATA_DIR, "train_source3.tsv"), separator="\t")
    df_targets = pl.concat([df_s2, df_s3])
    prep_targets = extract_record_features(df_targets, is_s1=False)

    blocker = MultiPathBlocker()
    all_candidates: Dict[str, Set[str]] = {}

    for country in prep_s1["country"].unique().to_list():
        country_s1 = prep_s1.filter(pl.col("country") == country)
        country_targets = prep_targets.filter(pl.col("country") == country)

        blocker.build_index_for_country(country_targets)
        cands = blocker.query_s1_for_country(country_s1)
        all_candidates.update(cands)

    # Calculate overall recall ceiling
    captured = 0
    total_cand_count = sum(len(c) for c in all_candidates.values())
    for sid, true_matches in gt_map.items():
        cands = all_candidates.get(sid, set())
        captured += len(true_matches.intersection(cands))

    recall = captured / total_true if total_true > 0 else 0.0
    avg_cands = total_cand_count / len(sample_s1_ids)

    print("\n" + "=" * 60)
    print("MULTI-PATH BLOCKER RESULTS:")
    print(f"Total S1 entities: {len(sample_s1_ids):,}")
    print(f"Total Candidates: {total_cand_count:,}")
    print(f"Average candidates per S1: {avg_cands:.1f}")
    print(f"True Matches: {total_true:,}")
    print(f"Captured True Matches: {captured:,}")
    print(f"Recall Ceiling: {recall * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
