"""
v8 generic matcher: GPU features + per-country label-free self-training.

Why self-training: countries label the same evidence differently (measured on
train: identical names are a match 41% of the time in US vs 21% in India), so
no fixed model is right for an unseen country such as France. At inference the
base model (trained on every labelled country) scores the new country, its
confident predictions become pseudo-labels, and the model is retrained on
source labels + pseudo-labels. No labels from the new country are used.

Leave-one-country-out (scripts/generalization_eval_v8.py): IN->US 0.889 -> 0.902,
US->IN 0.768 -> 0.768; average +0.007.
"""

import os
import pickle
import time
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import polars as pl
import torch
import xgboost as xgb

from blocking import MultiPathBlocker, extract_record_features
from gpu import require_cuda, xgb_gpu_params, xgb_predict_gpu
from gpu_features import FEATURE_NAMES_V8, build_country_vocab, encode_bank, features_v8

# v9 = v8 + multilingual semantic layer (semantic.py)
SEM_FEATURE_NAMES = ["name_sem_cos", "addr_sem_cos", "grp_name_sem_gap"]
FEATURE_NAMES_V9 = FEATURE_NAMES_V8 + SEM_FEATURE_NAMES
SEM_TOPK = 10   # measured: +0.9 pt recall US and India for ~6 extra candidates per S1


def feature_names_for(version: str) -> List[str]:
    return FEATURE_NAMES_V9 if version in ("v9", "v10") else FEATURE_NAMES_V8


def semantic_candidates(prep_s1: pl.DataFrame, prep_targets: pl.DataFrame, k: int = SEM_TOPK):
    """Top-k targets per S1 by multilingual name-embedding cosine (GPU, exact).

    Returns ({s1_id: set(target_id)}, {s1_id: name embedding row}, target-name embeddings, target ids).
    """
    from semantic import embed, topk_search
    t0 = time.time()
    t_ids = prep_targets["target_id"].to_list()
    s_ids = prep_s1["s1_id"].to_list()
    Et = embed(prep_targets["name_raw"].to_list())
    Es = embed(prep_s1["name_raw"].to_list())
    nn = topk_search(Es, Et, k)
    cands = {s: {t_ids[j] for j in nn[r]} for r, s in enumerate(s_ids)}
    log(f"  semantic top-{k}: embedded {len(t_ids):,} targets + {len(s_ids):,} S1, searched in {time.time() - t0:.0f}s")
    return cands, Es, Et


def semantic_features_into(prep_s1: pl.DataFrame, prep_targets: pl.DataFrame, s_ids: List[str], t_ids: List[str],
                           ia: np.ndarray, ib: np.ndarray, out: np.ndarray, Es_all=None, Et_all=None) -> None:
    """Writes the 3 semantic columns (after the v8 columns) for pairs (s_ids[ia], t_ids[ib])."""
    from semantic import embed, pair_cosine
    t0 = time.time()
    n8 = len(FEATURE_NAMES_V8)
    s_frame = _rows_in_order(prep_s1, "s1_id", s_ids)
    t_frame = _rows_in_order(prep_targets, "target_id", t_ids)
    if Es_all is not None and Et_all is not None:
        # reuse name embeddings computed for semantic blocking (rows aligned to prep frames)
        s_pos = dict(zip(prep_s1["s1_id"].to_list(), range(len(prep_s1))))
        t_pos = dict(zip(prep_targets["target_id"].to_list(), range(len(prep_targets))))
        En_s = Es_all[np.fromiter((s_pos[s] for s in s_ids), np.int64, len(s_ids))]
        En_t = Et_all[np.fromiter((t_pos[t] for t in t_ids), np.int64, len(t_ids))]
    else:
        En_s, En_t = embed(s_frame["name_raw"].to_list()), embed(t_frame["name_raw"].to_list())
    Ea_s, Ea_t = embed(s_frame["addr_raw"].to_list()), embed(t_frame["addr_raw"].to_list())
    name_cos = pair_cosine(En_s, En_t, ia, ib)
    gmax = np.full(len(s_ids), -2.0, dtype=np.float32)
    np.maximum.at(gmax, ia, name_cos)
    step = 5_000_000
    for lo in range(0, len(ia), step):
        hi = min(lo + step, len(ia))
        out[lo:hi, n8] = name_cos[lo:hi]
        out[lo:hi, n8 + 1] = pair_cosine(Ea_s, Ea_t, ia[lo:hi], ib[lo:hi])
        out[lo:hi, n8 + 2] = name_cos[lo:hi] - gmax[ia[lo:hi]]
    log(f"  semantic features for {len(ia):,} pairs in {time.time() - t0:.0f}s")

# Tuned with scripts/tune_v8.py (best on ID and both unseen-country protocols)
XGB_PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 7, "learning_rate": 0.05,
              "min_child_weight": 1, "colsample_bytree": 0.6, "subsample": 0.9, "random_state": 42}
MAX_ROUNDS, EARLY_STOP = 4000, 100
SELF_HI, SELF_LO = 0.97, 0.03
MAX_PSEUDO = 3_000_000
FEATURE_CHUNK = 1_000_000   # smaller blocks keep large countries (India) within GPU memory


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rows_in_order(prep: pl.DataFrame, id_col: str, ids: List[str]) -> pl.DataFrame:
    """Rows of prep for ids, in the order of ids (hash join; no per-block is_in scans)."""
    order = pl.DataFrame({id_col: ids, "_ord": np.arange(len(ids), dtype=np.int64)})
    return order.join(prep, on=id_col, how="left").sort("_ord").drop("_ord")


def features_into(prep_s1: pl.DataFrame, prep_targets: pl.DataFrame, s_ids: List[str], t_ids: List[str],
                  ia: np.ndarray, ib: np.ndarray, out: np.ndarray) -> None:
    """v8 features for pairs (s_ids[ia[k]], t_ids[ib[k]]) written into out (array or memmap).

    Pairs must be grouped by S1 (ia non-decreasing in groups). Vocabulary comes
    from the country's full target corpus (label-free). Work proceeds in blocks of
    whole S1 groups so GPU memory stays bounded; only the records a block needs
    are encoded.
    """
    dev = require_cuda()
    t0 = time.time()
    voc = build_country_vocab(prep_targets)
    t_voc = time.time() - t0
    n = len(ia)
    t_enc = t_gpu = 0.0
    lo = 0
    while lo < n:
        hi = min(lo + FEATURE_CHUNK, n)
        while hi < n and ia[hi] == ia[hi - 1]:
            hi += 1
        t1 = time.time()
        us, la = np.unique(ia[lo:hi], return_inverse=True)
        ut, lb = np.unique(ib[lo:hi], return_inverse=True)
        bs = encode_bank(_rows_in_order(prep_s1, "s1_id", [s_ids[i] for i in us]), voc, "s1_id", pattern_side=True)
        bt = encode_bank(_rows_in_order(prep_targets, "target_id", [t_ids[i] for i in ut]), voc, "target_id",
                         pattern_side=False)
        t2 = time.time()
        out[lo:hi, :len(FEATURE_NAMES_V8)] = features_v8(bs, bt, torch.as_tensor(la, device=dev),
                                                         torch.as_tensor(lb, device=dev)).cpu().numpy()
        t_enc += t2 - t1
        t_gpu += time.time() - t2
        del bs, bt
        torch.cuda.empty_cache()   # hand torch's cached blocks back (next block / XGBoost)
        lo = hi
    log(f"  {n:,} pairs: vocab {t_voc:.0f}s, encode {t_enc:.0f}s, GPU features {t_gpu:.1f}s "
        f"(total {time.time() - t0:.0f}s)")


def pair_arrays(cands: Dict[str, Set[str]]) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """Candidate map -> (s_ids, t_ids, ia, ib) int32 arrays grouped by S1, each group sorted by target id."""
    s_ids = sorted(cands)
    t_ids = sorted(set().union(*cands.values())) if cands else []
    t_index = {t: i for i, t in enumerate(t_ids)}
    sizes = np.fromiter((len(cands[s]) for s in s_ids), np.int64, len(s_ids))
    ia = np.repeat(np.arange(len(s_ids), dtype=np.int32), sizes)
    ib = np.fromiter((t_index[t] for s in s_ids for t in sorted(cands[s])), np.int32, int(sizes.sum()))
    return s_ids, t_ids, ia, ib


def country_features(prep_s1: pl.DataFrame, prep_targets: pl.DataFrame,
                     pairs: Sequence[Tuple[str, str]], version: str = "v8", sem=None) -> np.ndarray:
    """In-memory variant for (training-sized) lists of (s1_id, target_id) pairs grouped by S1.

    sem: optional (Es_all, Et_all) name embeddings aligned with prep_s1 / prep_targets rows.
    """
    s_ids = list(dict.fromkeys(s for s, _ in pairs))
    t_ids = sorted({t for _, t in pairs})
    si = {k: i for i, k in enumerate(s_ids)}
    ti = {k: i for i, k in enumerate(t_ids)}
    ia = np.fromiter((si[s] for s, _ in pairs), np.int32, len(pairs))
    ib = np.fromiter((ti[t] for _, t in pairs), np.int32, len(pairs))
    X = np.empty((len(pairs), len(feature_names_for(version))), dtype=np.float32)
    features_into(prep_s1, prep_targets, s_ids, t_ids, ia, ib, X)
    if version in ("v9", "v10"):
        semantic_features_into(prep_s1, prep_targets, s_ids, t_ids, ia, ib, X, *(sem or (None, None)))
    return X


def predict(booster, X: np.ndarray, chunk: int = 4_000_000) -> np.ndarray:
    return np.concatenate([xgb_predict_gpu(booster, X[i:i + chunk]).cpu().numpy()
                           for i in range(0, len(X), chunk)]) if len(X) else np.zeros(0, np.float32)


def train_booster(X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray, rounds: int = MAX_ROUNDS):
    dev = require_cuda()
    names = FEATURE_NAMES_V9 if X.shape[1] == len(FEATURE_NAMES_V9) else FEATURE_NAMES_V8
    torch.cuda.empty_cache()
    Xg, yg = torch.as_tensor(X, device=dev), torch.as_tensor(y, device=dev)
    tr = xgb.QuantileDMatrix(Xg, label=yg, feature_names=names)
    del Xg, yg
    Xg, yg = torch.as_tensor(Xv, device=dev), torch.as_tensor(yv, device=dev)
    va = xgb.QuantileDMatrix(Xg, label=yg, feature_names=names, ref=tr)
    del Xg, yg
    torch.cuda.empty_cache()   # QuantileDMatrix keeps its own compressed copy
    bst = xgb.train(xgb_gpu_params(XGB_PARAMS), tr, rounds, evals=[(va, "val")],
                    early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    return bst[: bst.best_iteration + 1]


def self_train(booster, src: Dict[str, np.ndarray], X_tgt: np.ndarray, p_tgt: np.ndarray,
               seed: int = 42, max_pseudo: int = MAX_PSEUDO):
    """Retrain on source labels + confident pseudo-labels of the unlabelled target country."""
    conf = np.nonzero((p_tgt >= SELF_HI) | (p_tgt <= SELF_LO))[0]
    if len(conf) > max_pseudo:
        conf = np.sort(np.random.default_rng(seed).choice(conf, max_pseudo, replace=False))
    if len(conf) == 0:
        return booster
    y_p = (p_tgt[conf] >= SELF_HI).astype(np.float32)
    log(f"  self-training on {len(conf):,} pseudo-labelled pairs ({int(y_p.sum()):,} positive)")
    X = np.concatenate([src["X"], X_tgt[conf]])
    y = np.concatenate([src["y"], y_p])
    return train_booster(X, y, src["Xv"], src["yv"])


def judge_scores(judge, ck, X, p, groups, s1_rows_edges, chunk_rows: int = 20000) -> np.ndarray:
    """Replace v9 scores of each S1's top-m candidates with the judge's calibrated probabilities.

    s1_rows_edges(lo, hi) -> edges (hi-lo, m, m, Fe) for S1 rows lo..hi (computed lazily so large
    countries never hold every edge tensor in memory).
    """
    from judge import make_judge_data, predict_judge
    out = p.copy()
    for lo in range(0, len(groups), chunk_rows):
        hi = min(lo + chunk_rows, len(groups))
        g = groups[lo:hi]
        data = make_judge_data(g, X, s1_rows_edges(lo, hi), p, ck["mu"], ck["sd"])
        pj = predict_judge(judge, data)["p"]
        ok = g >= 0
        out[g[ok]] = pj[ok]
    return out


def _train_judge_stage(model_dir, pairs, X, y, gt_map, tr_ids, va_ids, is_va, p_va, groups, edges, group_s1, seed):
    """v10: out-of-fold v9 scores -> Jev-style set judge -> calibrated threshold on validation."""
    from decision import assign_threshold
    from judge import JudgeConfig, make_judge_data, save_judge, train_judge
    from metrics import compute_macro_f05

    t0 = time.time()
    # out-of-fold v9 scores on training entities (the judge must not see in-sample scores)
    p = np.zeros(len(pairs), dtype=np.float32)
    p[is_va] = p_va
    tr_list = sorted(tr_ids)
    np.random.default_rng(seed).shuffle(tr_list)
    fold_of = {s: k % 5 for k, s in enumerate(tr_list)}
    pair_fold = np.array([fold_of.get(s, -1) for s, _ in pairs])
    for f in range(5):
        held, rest = pair_fold == f, (pair_fold >= 0) & (pair_fold != f)
        b = train_booster(X[rest], y[rest], X[is_va], y[is_va])
        p[held] = predict(b, X[held])
    log(f"  out-of-fold v9 scores in {time.time() - t0:.0f}s")

    va_set = set(va_ids)
    rows_va = np.array([i for i, s in enumerate(group_s1) if s in va_set])
    rows_tr = np.array([i for i, s in enumerate(group_s1) if s not in va_set])
    gtr = groups[rows_tr]
    mu = X[gtr[gtr >= 0]].mean(0)
    sd = X[gtr[gtr >= 0]].std(0) + 1e-6
    dtr = make_judge_data(gtr, X, edges[rows_tr], p, mu, sd, y)
    dva = make_judge_data(groups[rows_va], X, edges[rows_va], p, mu, sd, y)
    judge = train_judge(dtr, dva, JudgeConfig(n_node_features=dtr.nodes.shape[-1]), log=log)

    from judge import predict_judge
    s = p.copy()
    gv = groups[rows_va]
    ok = gv >= 0
    s[gv[ok]] = predict_judge(judge, dva)["p"][ok]
    va_pairs = [q for q, v in zip(pairs, is_va) if v]
    va_gt = {k: gt_map[k] for k in va_ids}
    scores = {t: compute_macro_f05(va_gt, assign_threshold(va_pairs, s[is_va], t))["macro_f05"]
              for t in [round(x, 2) for x in np.arange(0.40, 0.91, 0.05)]}
    tau = max(scores, key=scores.get)
    log(f"v10 judge: validation Macro F0.5 {scores[tau]:.4f} at tau {tau} ({time.time() - t0:.0f}s)")
    fname = "judge_v10.pt"
    save_judge(os.path.join(model_dir, fname), judge, mu, sd, tau, {"val_macro_f05": scores[tau]})
    return {"tau": tau, "val_macro_f05": scores[tau], "file": fname}


# --------------------------------------------------------------- training
def train_v8(data_dir: str, model_dir: str, per_country: int = 20000, seed: int = 42, version: str = "v8") -> dict:
    """Train the base model on every labelled country; save model, config and source tensors.

    version "v9" adds semantic top-k candidates and the semantic features. Each
    version writes its own model/source files; config.pkl (the active config) is
    backed up as config_<previous version>_backup.pkl before being replaced.
    """
    from decision import assign_threshold
    from metrics import compute_macro_f05
    from sklearn.model_selection import train_test_split

    require_cuda()
    t_start = time.time()
    gt = pl.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), separator="\t")
    s1 = pl.read_csv(os.path.join(data_dir, "train_source1.tsv"), separator="\t")
    tg = pl.concat([pl.read_csv(os.path.join(data_dir, f), separator="\t")
                    for f in ("train_source2.tsv", "train_source3.tsv")])
    gt = gt.join(s1.select(["entity_id", "country"]), left_on="source1_entity_id", right_on="entity_id")
    countries = sorted(gt["country"].unique().to_list())
    sampled = pl.concat([gt.filter(pl.col("country") == c).sample(
        n=min(per_country, gt.filter(pl.col("country") == c).height), seed=seed) for c in countries])
    gt_map: Dict[str, Set[str]] = {s: {t.strip() for t in (m or "").split(",") if t.strip()}
                                   for s, m in sampled.select(["source1_entity_id", "matched_entity_ids"]).iter_rows()}
    log(f"training countries {countries}: {len(gt_map):,} S1 entities")

    prep_s1_all = extract_record_features(s1.filter(pl.col("entity_id").is_in(list(gt_map))), is_s1=True)
    prep_t_all = extract_record_features(tg)
    pairs: List[Tuple[str, str]] = []
    feats: List[np.ndarray] = []
    groups: List[np.ndarray] = []
    edges: List[np.ndarray] = []
    group_s1: List[str] = []
    for country in countries:
        ps = prep_s1_all.filter(pl.col("country") == country)
        pt = prep_t_all.filter(pl.col("country") == country)
        blocker = MultiPathBlocker()
        blocker.build_index_for_country(pt)
        cands = blocker.query_s1_for_country(ps)
        del blocker
        sem = None
        if version in ("v9", "v10"):
            sem_cands, Es, Et = semantic_candidates(ps, pt)
            for s, ts in sem_cands.items():
                cands.setdefault(s, set()).update(ts)
            sem = (Es, Et)
        cp = [(s, t) for s in sorted(cands) for t in sorted(cands[s])]
        hit = sum(len(gt_map[s] & cands[s]) for s in cands)
        true = sum(len(gt_map[s]) for s in cands)
        log(f"[{country}] {len(cp):,} candidate pairs, blocking recall {hit / max(true, 1):.4f}")
        Xc = country_features(ps, pt, cp, version, sem)
        if version == "v10":
            # graph-consistency edges between each S1's top-m candidates (label-free ranking)
            from judge import topm_groups, tt_edge_features
            s_local = {s: i for i, s in enumerate(dict.fromkeys(s for s, _ in cp))}
            ia_c = np.fromiter((s_local[s] for s, _ in cp), np.int64, len(cp))
            g = topm_groups(ia_c, Xc, feature_names_for(version))
            t_list = pt["target_id"].to_list()
            t_pos = {tid: i for i, tid in enumerate(t_list)}
            gt_idx = np.where(g >= 0, np.vectorize(lambda q: t_pos[cp[q][1]] if q >= 0 else -1)(g), -1)
            t0e = time.time()
            edges.append(tt_edge_features(pt, build_country_vocab(pt), gt_idx, t_list, sem[1]))
            log(f"[{country}] graph edges for {len(g):,} S1 in {time.time() - t0e:.0f}s")
            groups.append(np.where(g >= 0, g + len(pairs), -1))
            group_s1 += list(s_local)
        feats.append(Xc)
        del sem
        pairs += cp
    X = np.concatenate(feats)
    y = np.array([t in gt_map[s] for s, t in pairs], dtype=np.float32)

    tr_ids, va_ids = train_test_split(sorted(gt_map), test_size=0.15, random_state=seed)
    va_set = set(va_ids)
    is_va = np.array([s in va_set for s, _ in pairs])
    bst = train_booster(X[~is_va], y[~is_va], X[is_va], y[is_va])
    p_va = predict(bst, X[is_va])
    va_pairs = [q for q, v in zip(pairs, is_va) if v]
    va_gt = {s: gt_map[s] for s in va_ids}
    scores = {t: compute_macro_f05(va_gt, assign_threshold(va_pairs, p_va, t))["macro_f05"]
              for t in [round(x, 2) for x in np.arange(0.40, 0.91, 0.05)]}
    tau = max(scores, key=scores.get)
    log(f"validation Macro F0.5 {scores[tau]:.4f} at tau {tau} ({bst.num_boosted_rounds()} trees)")
    judge_info = {}
    if version == "v10":
        judge_info = _train_judge_stage(model_dir, pairs, X, y, gt_map, tr_ids, va_ids, is_va, p_va,
                                        np.concatenate(groups), np.concatenate(edges), group_s1, seed)

    os.makedirs(model_dir, exist_ok=True)
    model_file = f"xgboost_matcher_{version}.json"
    bst.save_model(os.path.join(model_dir, model_file))
    torch.save({"X": X[~is_va], "y": y[~is_va], "Xv": X[is_va], "yv": y[is_va]},
               os.path.join(model_dir, f"{version}_source.pt"))
    config = {"feature_version": version, "feature_names": feature_names_for(version), "model_type": "xgboost",
              "model_file": model_file, "best_threshold": tau, "decision": "threshold",
              "self_train": True, "val_macro_f05": scores[tau], "trained_countries": countries,
              "trained_per_country": per_country}
    if judge_info:
        # v10: final decisions come from the judge; keep the v9-stage numbers for reference
        config.update({"v9_threshold": tau, "v9_val_macro_f05": scores[tau], "best_threshold": judge_info["tau"],
                       "val_macro_f05": judge_info["val_macro_f05"], "judge_file": judge_info["file"]})
    cfg_path = os.path.join(model_dir, "config.pkl")
    if os.path.exists(cfg_path):
        with open(cfg_path, "rb") as f:
            prev = pickle.load(f)
        backup = os.path.join(model_dir, f"config_{prev.get('feature_version', 'old')}_backup.pkl")
        with open(backup, "wb") as f:
            pickle.dump(prev, f)
        log(f"previous active config backed up to {os.path.basename(backup)}")
    with open(cfg_path, "wb") as f:
        pickle.dump(config, f)
    with open(os.path.join(model_dir, f"config_{version}.pkl"), "wb") as f:
        pickle.dump(config, f)
    log(f"saved {version} model, source tensors and config in {time.time() - t_start:.0f}s")
    return config
